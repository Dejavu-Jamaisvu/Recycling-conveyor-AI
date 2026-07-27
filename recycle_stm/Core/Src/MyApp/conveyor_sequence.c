#include "conveyor_sequence.h"
#include "servo.h"
#include "main.h"

static ConveyorSeqState state = SEQ_IDLE;
static uint32_t stop_start_time = 0;

static uint8_t IsObjectDetected(void)
{
    return (HAL_GPIO_ReadPin(SENSOR_GPIO_Port, SENSOR_Pin) == GPIO_PIN_RESET);
}

static void SendLineSignal(uint8_t stop)
{
    HAL_GPIO_WritePin(ARD_SIG_GPIO_Port, ARD_SIG_Pin,
                       stop ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void ConveyorSequence_Init(void)
{
    SendLineSignal(0);   // 시작은 "가" 상태
    state = SEQ_IDLE;
}

void ConveyorSequence_Update(void)
{
    switch (state)
    {
    case SEQ_IDLE:
        state = SEQ_MOVING;
        break;

    case SEQ_MOVING:
        if (IsObjectDetected())
        {
            state = SEQ_OBJECT_DETECTED;
        }
        break;

    case SEQ_OBJECT_DETECTED:
        SendLineSignal(1);       // 아두이노에게 "멈춰" 신호
        stop_start_time = HAL_GetTick();
        state = SEQ_ACTUATING;   // ★ Jetson 결과 기다리는 단계 생략, 바로 서보 동작으로
        break;

    case SEQ_ACTUATING:
    {
        static uint8_t toggle = 0;
        Servo_SetAngle(toggle ? 0 : 180);
        toggle = !toggle;
        HAL_Delay(1500);
        SendLineSignal(0);
        state = SEQ_MOVING;
        break;
    }

    default:
        state = SEQ_MOVING;
        break;
    }
}
