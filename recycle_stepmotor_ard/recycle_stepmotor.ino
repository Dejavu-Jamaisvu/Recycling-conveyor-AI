const int PUL_PIN = 7;
const int DIR_PIN = 6;
const int ENA_PIN = 5;
const int SIGNAL_PIN = 4;   // STM32의 ARD_SIG 핀과 연결

const int PULSE_DELAY_US = 100;  // 예제 코드 기준으로 맞춤 (더 빠르게 돔)

void setup() {
  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENA_PIN, OUTPUT);
  pinMode(SIGNAL_PIN, INPUT);

  digitalWrite(DIR_PIN, LOW);    // 방향 (반대로 돌면 HIGH로 변경)
  digitalWrite(ENA_PIN, HIGH);   // ★ 활성화 — LOW에서 HIGH로 수정
}

void loop() {
  bool stopSignal = digitalRead(SIGNAL_PIN);  // STM32: HIGH=멈춰, LOW=가

  if (stopSignal == HIGH) {
    digitalWrite(PUL_PIN, HIGH);
    delayMicroseconds(PULSE_DELAY_US);
    digitalWrite(PUL_PIN, LOW);
    delayMicroseconds(PULSE_DELAY_US);
  }
}