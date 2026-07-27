#include "ap.h"
#include "conveyor_sequence.h"
#include "servo.h"

void AP_Init(void)
{
	Servo_Init();          // 추가
	ConveyorSequence_Init();
}

void AP_Run(void)
{
    while (1)
    {
        ConveyorSequence_Update();
    }
}
