#ifndef MYAPP_CONVEYOR_SEQUENCE_H
#define MYAPP_CONVEYOR_SEQUENCE_H

#include "main.h"

/* =========================================================
 * 전체 시퀀스 상태머신
 *
 *  SEQ_MOVING      : 벨트 ON, IR 감지 대기
 *  SEQ_WAIT_RESULT : 벨트 정지, Jetson 추론 결과 대기
 *  SEQ_SORTING     : 결과대로 게이트 세팅 후 벨트 재가동,
 *                    물체 통과/낙하 시간만큼 대기
 *
 * 통신 프로토콜 (USART1 = 핀헤더 배선, 115200 8N1)
 *  STM32 → Jetson : "TRIGGER\n"
 *  Jetson → STM32 : "CLASS:metal|paper|plastic|unknown\n"
 * ========================================================= */

void ConveyorSequence_Init(void);
void ConveyorSequence_Run(void);   /* while(1)에서 계속 호출 */

#endif /* MYAPP_CONVEYOR_SEQUENCE_H */
