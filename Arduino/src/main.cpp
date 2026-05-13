#include <Arduino.h>

constexpr uint8_t RESTART_BUTTON_PIN = 4;
constexpr uint8_t PAUSE_BUTTON_PIN = 7;
constexpr unsigned long DEBOUNCE_DELAY_MS = 20;

bool lastRestartButtonState = HIGH;
bool lastPauseButtonState = HIGH;

void handleButtonPress(uint8_t pin, bool currentState, bool &lastState, const char *command) {
  if (currentState == LOW && lastState == HIGH) {
    delay(DEBOUNCE_DELAY_MS);
    if (digitalRead(pin) == LOW) {
      Serial.println(command);
      while (digitalRead(pin) == LOW) {
        delay(1);
      }
    }
  }

  lastState = currentState;
}

void setup() {
  Serial.begin(9600);

  pinMode(RESTART_BUTTON_PIN, INPUT_PULLUP);
  pinMode(PAUSE_BUTTON_PIN, INPUT_PULLUP);

  lastRestartButtonState = digitalRead(RESTART_BUTTON_PIN);
  lastPauseButtonState = digitalRead(PAUSE_BUTTON_PIN);
}

void loop() {
  bool restartButtonState = digitalRead(RESTART_BUTTON_PIN);
  bool pauseButtonState = digitalRead(PAUSE_BUTTON_PIN);

  handleButtonPress(RESTART_BUTTON_PIN, restartButtonState, lastRestartButtonState, "RESTART_INFERENCE");
  handleButtonPress(PAUSE_BUTTON_PIN, pauseButtonState, lastPauseButtonState, "TOGGLE_INFER_PAUSED");
}
