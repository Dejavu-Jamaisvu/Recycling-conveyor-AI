#include "conveyor_sequence.h"
#include "servo.h"
#include <string.h>

/* =========================================================
 * Jetson 통신: USART1 (PA9/PA10 ↔ Jetson J41 핀헤더, /dev/ttyTHS1)
 *  - 배선: Jetson 핀8(TXD)→PA10, 핀10(RXD)→PA9, 핀6 GND→GND
 *  - Baudrate 115200 (usart.c의 huart1 설정과 Jetson 스크립트
 *    SERIAL_BAUDRATE가 동일해야 함)
 *
 * 프로토콜 (팀원 스크립트 4-2와 동일)
 *  STM32 → Jetson : "TRIGGER\n"
 *  Jetson → STM32 : "CLASS:metal\n" / "CLASS:paper\n"
 *                   "CLASS:plastic\n" / "CLASS:unknown\n"
 * ========================================================= */

extern UART_HandleTypeDef huart1;

/* ---------- 핀 정의 (CubeMX 라벨 사용) ---------- */
#define IR_PORT        SENSOR_GPIO_Port    /* MH-Sensor, Active LOW */
#define IR_PIN         SENSOR_Pin
#define ARD_PORT       ARD_SIG_GPIO_Port   /* 아두이노 정지/재가동 신호 */
#define ARD_PIN        ARD_SIG_Pin

/* ---------- 타이밍 (실물 보고 조정) ---------- */
#define RESULT_TIMEOUT_MS  30000     /* 추론 응답 최대 대기 */
#define SORT_PLASTIC_MS   7000   /* 첫 게이트 도착 시간 + 여유 */
#define SORT_PAPER_MS     17000   /* 두 번째 게이트 도착 + 여유 */
#define SORT_METAL_MS     25000   /* 끝까지 가서 낙하 완료 */
#define IR_REARM_MS        1500     /* 재가동 직후 IR 재감지 무시 시간 */
#define MAX_RETRY   2      /* unknown일 때 재촬영 횟수 */

/* ---------- 분류 결과 ---------- */
#define CLASS_METAL     0
#define CLASS_PAPER     1
#define CLASS_PLASTIC   2
#define CLASS_UNKNOWN   3

#define JOG_BACK_MS  1000

static uint8_t retry_count = 0;
static uint32_t sort_ms = SORT_METAL_MS;   /* 이번 사이클의 대기 시간 */

typedef enum {
    SEQ_MOVING = 0,
    SEQ_WAIT_RESULT,
    SEQ_SORTING,
	SEQ_JOG_BACK,

} SeqState;

static SeqState        state = SEQ_MOVING;
static uint32_t        t_ref = 0;

/* ---- UART 라인 수신용 ---- */
#define RX_BUF_LEN 32
static uint8_t         rx_byte;
static char            rx_line[RX_BUF_LEN];
static uint8_t         rx_pos = 0;
static volatile int8_t rx_result = -1;   /* -1: 아직 없음 */
static uint8_t ir_armed = 1;   /* IR이 비워진(HIGH) 걸 본 뒤에만 감지 허용 */

/* 아두이노 신호: run=1 이면 벨트 ON, 0이면 정지
 * (아두이노 쪽 판정 로직과 반대면 SET/RESET만 바꾸면 됨) */
static void Belt(uint8_t run)
{
    HAL_GPIO_WritePin(ARD_PORT, ARD_PIN,
                      run ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void BeltDir(uint8_t reverse)   /* Belt() 함수 아래에 */
{
    HAL_GPIO_WritePin(DIR_SIG_GPIO_Port, DIR_SIG_Pin,
                      reverse ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void RequestCapture(void)
{
    static const char msg[] = "TRIGGER\n";
    rx_result = -1;
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, sizeof(msg) - 1, 100);
}

/* 결과에 따라 게이트 세팅
 *  metal   : 1열림, 2열림  → 끝까지 감
 *  paper   : 1열림, 2사선  → 두 번째에서 낙하
 *  plastic : 1사선, 2사선  → 첫 번째에서 낙하
 *  unknown : 1사선, 2사선  → 일단 첫 번째로 (원하면 변경) */
static void ApplyGates(int8_t cls)
{
    switch (cls) {
    case CLASS_METAL:   Servo_SetGates(0, 0); break;
    case CLASS_PAPER:   Servo_SetGates(0, 1); break;
    case CLASS_PLASTIC: Servo_SetGates(1, 0); break;
    case CLASS_UNKNOWN:
    default:            Servo_SetGates(0, 1); break;
    }
}

void ConveyorSequence_Init(void)
{
    Servo_Init();                                  /* 게이트 사선 초기화 */
    HAL_UART_Receive_IT(&huart1, &rx_byte, 1);     /* 수신 인터럽트 시작 */
    Belt(1);                                       /* 벨트 가동 */
    t_ref = HAL_GetTick();
    state = SEQ_MOVING;
}

void ConveyorSequence_Run(void)
{
    uint32_t now = HAL_GetTick();

    switch (state) {

    case SEQ_MOVING:
        /* 재가동 직후에는 IR 무시 (같은 물체 재트리거 방지) */
        if (now - t_ref < IR_REARM_MS) break;

        if (HAL_GPIO_ReadPin(IR_PORT, IR_PIN) == GPIO_PIN_SET) {
                ir_armed = 1;                          /* 센서 앞이 비었음 → 무장 */
            }
        /* 무장 완료 = 투입 가능 → LED 켜기 */
           HAL_GPIO_WritePin(USER_LED_GPIO_Port, USER_LED_Pin,
                             ir_armed ? GPIO_PIN_SET : GPIO_PIN_RESET);

           if (ir_armed && HAL_GPIO_ReadPin(IR_PORT, IR_PIN) == GPIO_PIN_RESET) {
               ir_armed = 0;
               HAL_GPIO_WritePin(USER_LED_GPIO_Port, USER_LED_Pin, GPIO_PIN_RESET); /* 처리 시작 → 끔 */
               Belt(0);
               HAL_Delay(300);
               RequestCapture();
               HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET); /* 디버그 */
               t_ref = HAL_GetTick();
               retry_count = 0;
               state = SEQ_WAIT_RESULT;
        }
        break;

    case SEQ_WAIT_RESULT:
           if (rx_result >= 0) {
               if (rx_result == CLASS_UNKNOWN && retry_count < MAX_RETRY) {
            	      retry_count++;
            	      BeltDir(1);              /* 역방향 */
            	      Belt(1);                 /* 후진 시작 */
            	      t_ref = HAL_GetTick();
            	      state = SEQ_JOG_BACK;
            	      break;
               }
               HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
               ApplyGates(rx_result);
               switch (rx_result) {
               case CLASS_PLASTIC: sort_ms = SORT_PLASTIC_MS; break;
               case CLASS_PAPER:   sort_ms = SORT_PAPER_MS;   break;
               case CLASS_METAL:   sort_ms = SORT_METAL_MS;   break;
               default:            sort_ms = SORT_PLASTIC_MS; break;   /* unknown은 첫 게이트 */
               }
               HAL_Delay(400);
               Belt(1);
               t_ref = HAL_GetTick();
               state = SEQ_SORTING;
           }
        else if (now - t_ref > RESULT_TIMEOUT_MS) {
            /* 응답 없음 → unknown 취급하고 계속 진행 */
            HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
            ApplyGates(CLASS_UNKNOWN);
            Belt(1);
            t_ref = HAL_GetTick();
            state = SEQ_SORTING;
        }
        break;

    case SEQ_SORTING:
        if (now - t_ref > sort_ms) {
            Servo_AllBlock();     /* 게이트 원위치(사선) */
            t_ref = HAL_GetTick();
            state = SEQ_MOVING;
        }
        break;
    case SEQ_JOG_BACK:
         if (now - t_ref > JOG_BACK_MS) {
             Belt(0);
             BeltDir(0);                          /* 정방향 복귀 (멈춘 상태에서 전환) */
             HAL_Delay(200);
             Belt(1);                             /* 다시 전진 */
             t_ref = HAL_GetTick() - IR_REARM_MS; /* ★ 재감지 잠금 건너뛰기 */
             state = SEQ_MOVING;
         }
         break;


    }


}

/* 수신된 한 줄 파싱: "CLASS:xxx" */
static void ParseLine(const char *line)
{
    if (strncmp(line, "CLASS:", 6) != 0) return;   /* 그 외 메시지 무시 */
    const char *cls = line + 6;

    if      (strcmp(cls, "metal")   == 0) rx_result = CLASS_METAL;
    else if (strcmp(cls, "paper")   == 0) rx_result = CLASS_PAPER;
    else if (strcmp(cls, "plastic") == 0) rx_result = CLASS_PLASTIC;
    else                                  rx_result = CLASS_UNKNOWN;
}

/* USART1 수신 인터럽트: 1바이트씩 모아서 '\n' 기준으로 라인 완성 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        if (rx_byte == '\n' || rx_byte == '\r') {
            if (rx_pos > 0) {
                rx_line[rx_pos] = '\0';
                ParseLine(rx_line);
                rx_pos = 0;
            }
        }
        else if (rx_pos < RX_BUF_LEN - 1) {
            rx_line[rx_pos++] = (char)rx_byte;
        }
        else {
            rx_pos = 0;   /* 버퍼 초과 → 라인 버림 */
        }
        HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
    }
}
