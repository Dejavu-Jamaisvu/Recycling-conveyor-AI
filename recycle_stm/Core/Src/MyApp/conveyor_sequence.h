#ifndef CONVEYOR_SEQUENCE_H
#define CONVEYOR_SEQUENCE_H

typedef enum {
    SEQ_IDLE,
    SEQ_MOVING,
    SEQ_OBJECT_DETECTED,
    SEQ_ACTUATING,
} ConveyorSeqState;

void ConveyorSequence_Init(void);
void ConveyorSequence_Update(void);

#endif
