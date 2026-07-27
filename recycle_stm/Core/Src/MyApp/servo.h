#ifndef SERVO_H
#define SERVO_H

#include <stdint.h>

void Servo_Init(void);
void Servo_SetAngle(uint16_t angle_deg);
void Servo_MoveToClass(uint8_t class_id);

#endif
