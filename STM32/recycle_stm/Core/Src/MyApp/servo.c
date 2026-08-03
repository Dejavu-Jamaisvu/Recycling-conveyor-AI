#include "servo.h"

extern TIM_HandleTypeDef htim11;  /* 서보1: PB9  */
extern TIM_HandleTypeDef htim3;   /* 서보2: PA6  */

/* 각도(0~180) → 펄스폭(us). SG90 기준 500~2500us
 * ARR=19999, 1tick=1us 이므로 CCR에 그대로 us 값을 씀 */
static uint16_t AngleToPulse(uint8_t angle_deg)
{
    if (angle_deg > 180) angle_deg = 180;
    return (uint16_t)(500 + ((uint32_t)angle_deg * 2000) / 180);
}

void Servo_Init(void)
{
    HAL_TIM_PWM_Start(&htim11, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim3,  TIM_CHANNEL_1);
    Servo_AllBlock();          /* 기본 상태: 둘 다 사선(막힘) */
    HAL_Delay(500);            /* 초기 위치 도달 대기 */
}

void Servo1_SetAngle(uint8_t angle_deg)
{
    __HAL_TIM_SET_COMPARE(&htim11, TIM_CHANNEL_1, AngleToPulse(angle_deg));
}

void Servo2_SetAngle(uint8_t angle_deg)
{
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, AngleToPulse(angle_deg));
}

void Servo_SetGates(uint8_t servo1_open, uint8_t servo2_open)
{
    Servo1_SetAngle(servo1_open ? SERVO1_OPEN_ANGLE : SERVO1_BLOCK_ANGLE);
    Servo2_SetAngle(servo2_open ? SERVO2_OPEN_ANGLE : SERVO2_BLOCK_ANGLE);
}

void Servo_AllBlock(void)
{
    Servo_SetGates(0, 0);
}
