#include "servo.h"
#include "tim.h"

void Servo_Init(void)
{
    HAL_TIM_PWM_Start(&htim11, TIM_CHANNEL_1);
}

void Servo_SetAngle(uint16_t angle_deg)  // 0~180
{
    uint32_t pulse_us = 1000 + (angle_deg * 1000 / 180);
    __HAL_TIM_SET_COMPARE(&htim11, TIM_CHANNEL_1, pulse_us);
}

void Servo_MoveToClass(uint8_t class_id)
{
    switch (class_id)
    {
    case 1: Servo_SetAngle(0);   break;  // metal
    case 2: Servo_SetAngle(90);  break;  // paper
    case 3: Servo_SetAngle(180); break;  // plastic
    }
}
