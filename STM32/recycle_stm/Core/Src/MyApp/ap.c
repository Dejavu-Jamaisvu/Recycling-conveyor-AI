#include "ap.h"
#include "conveyor_sequence.h"

/* main.c 흐름:
 *   MX_GPIO_Init();
 *   MX_USART2_UART_Init();
 *   MX_USART1_UART_Init();   ← CubeMX 재생성 후 추가돼야 함
 *   MX_TIM11_Init();
 *   MX_TIM3_Init();          ← CubeMX 재생성 후 추가돼야 함
 *   AP_Init();
 *   while(1) AP_Run();
 */

void AP_Init(void)
{
    ConveyorSequence_Init();
}

void AP_Run(void)
{
    ConveyorSequence_Run();
}
