#ifndef MYAPP_SERVO_H
#define MYAPP_SERVO_H

#include "main.h"

/* =========================================================
 * 서보 게이트 제어 (2개)
 *  - 서보1: TIM11_CH1 (PB9)  ← 기존 그대로
 *  - 서보2: TIM3_CH1  (PA6)  ← CubeMX에서 새로 추가 필요
 *  두 타이머 모두 Prescaler=83, ARR=19999 (50Hz, 1tick=1us)
 * ========================================================= */

/* ▼▼▼ 실물 보고 캘리브레이션 필요한 값들 ▼▼▼ */
#define SERVO1_BLOCK_ANGLE   90  /* 서보1 사선(막힘) 각도 */
#define SERVO1_OPEN_ANGLE    180    /* 서보1 길 열림 각도    */
#define SERVO2_BLOCK_ANGLE   90   /* 서보2 사선(막힘) 각도 */
#define SERVO2_OPEN_ANGLE    0    /* 서보2 길 열림 각도    */
/* ▲▲▲ 조립 방향에 따라 0/45가 반대일 수 있음 ▲▲▲ */

void Servo_Init(void);
void Servo1_SetAngle(uint8_t angle_deg);
void Servo2_SetAngle(uint8_t angle_deg);

/* open=1이면 길 열림, open=0이면 사선(막힘) */
void Servo_SetGates(uint8_t servo1_open, uint8_t servo2_open);
void Servo_AllBlock(void);

#endif /* MYAPP_SERVO_H */
