const int PUL_PIN = 7;
const int DIR_PIN = 6;
const int ENA_PIN = 5;
const int SIGNAL_PIN = 4;   // STM32 ARD_SIG: HIGH=가동, LOW=정지
Search

const int DIRSIG_PIN = 3;   // STM32 DIR_SIG: LOW=정방향, HIGH=역방향

const int PULSE_DELAY_US = 100;

void setup() {
  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENA_PIN, OUTPUT);
  pinMode(SIGNAL_PIN, INPUT);
  pinMode(DIRSIG_PIN, INPUT);

  digitalWrite(ENA_PIN, HIGH);   // 드라이버 활성화
}

void loop() {
  // STM32가 지시한 방향 반영 (정방향이 반대로 돌면 HIGH/LOW 바꾸기)
  digitalWrite(DIR_PIN, digitalRead(DIRSIG_PIN) ? HIGH : LOW);

  // 가동 신호일 때만 펄스
  if (digitalRead(SIGNAL_PIN) == HIGH) {
    digitalWrite(PUL_PIN, HIGH);
    delayMicroseconds(PULSE_DELAY_US);
    digitalWrite(PUL_PIN, LOW);
    delayMicroseconds(PULSE_DELAY_US);
  }
}