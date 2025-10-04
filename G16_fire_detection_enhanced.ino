/*
 * G16 Fire Detection System - Enhanced Version
 * ESP32-based fire detection with Blynk IoT integration
 * Features:
 * - Fire sensor detection
 * - Buzzer alarm
 * - LED indicators
 * - Blynk cloud notifications
 * - Email alerts
 * - Real-time monitoring
 */

#define BLYNK_TEMPLATE_ID "TMPLkOtFt5y-"
#define BLYNK_TEMPLATE_NAME "fdc98"
#define BLYNK_AUTH_TOKEN "WyHXrQ1dtwD70iYLRCqTSvcKJ-OYpCC2"
#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

// Network Configuration
char auth[] = "WyHXrQ1dtwD70iYLRCqTSvcKJ-OYpCC2";
char ssid[] = "abcd";        // WiFi SSID
char pass[] = "zxcvbnm11";   // WiFi Password

BlynkTimer timer;

// Pin Definitions
#define FIRE_SENSOR_PIN 18   // Fire sensor input pin
#define BUZZER_PIN 14        // Buzzer output pin
#define RED_LED_PIN 2        // Red LED for fire alert
#define GREEN_LED_PIN 19     // Green LED for safe status

// Variables
int fireVal = 0;
bool fireDetected = false;
unsigned long lastFireAlert = 0;
const unsigned long ALERT_COOLDOWN = 10000; // 10 seconds cooldown

WidgetLED led(V1);           // Blynk LED widget

void setup() 
{
  // Initialize serial communication
  Serial.begin(9600);
  Serial.println("G16 Fire Detection System Starting...");
  
  // Configure pins
  pinMode(FIRE_SENSOR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  
  // Initialize LEDs
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, HIGH);
  
  // Connect to Blynk
  Serial.println("Connecting to Blynk...");
  Blynk.begin(auth, ssid, pass);
  
  // Wait for connection
  delay(2000);
  
  // Set up timer for sensor reading (every 500ms)
  timer.setInterval(500L, checkFireSensor);
  
  Serial.println("System initialized successfully!");
}

void loop() 
{
  Blynk.run();
  timer.run();
}

void checkFireSensor()
{
  // Read fire sensor value
  fireVal = digitalRead(FIRE_SENSOR_PIN);
   
  if (fireVal == LOW) // Fire detected (sensor is active low)
  {
    handleFireDetected();
  }
  else
  {
    handleNoFire();
  }  
}

void handleFireDetected()
{
  // Check if enough time has passed since last alert
  if (millis() - lastFireAlert > ALERT_COOLDOWN)
  {
    // Activate alarm system
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(GREEN_LED_PIN, LOW);
    
    // Update Blynk interface
    Blynk.virtualWrite(V0, 1);  // Send fire status to Blynk
    led.on();                   // Turn on Blynk LED
    
    // Send notifications
    Blynk.notify("🚨 FIRE DETECTED! 🚨");
    Blynk.email("9663890904u@gmail.com", "Fire Alert", "Fire detected in the monitored area!");
    Blynk.logEvent("fire_alert", "Fire detected at sensor location");
    
    // Serial output
    Serial.println("🚨 FIRE DETECTED!! 🚨");
    Serial.println("Status: DANGER - Fire detected in monitored area");
    Serial.println("Actions: Alarm activated, notifications sent");
    
    // Update last alert time
    lastFireAlert = millis();
    fireDetected = true;
  }
}

void handleNoFire()
{
  // Deactivate alarm system
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, HIGH);
  
  // Update Blynk interface
  Blynk.virtualWrite(V0, 0);  // Send safe status to Blynk
  led.off();                  // Turn off Blynk LED
  
  // Clear fire detected status
  if (fireDetected)
  {
    Blynk.notify("✅ Area safe - No fire detected");
    Serial.println("✅ Area cleared - No fire detected");
    fireDetected = false;
  }
  
  Serial.println("✅ NO FIRE DETECTED - Area safe");
}

// Blynk connection status handler
BLYNK_CONNECTED() 
{
  Serial.println("Connected to Blynk!");
  Blynk.syncVirtual(V0);
}

// Blynk virtual pin handler for manual control
BLYNK_WRITE(V0) 
{
  int value = param.asInt();
  if (value == 1)
  {
    Serial.println("Manual fire alert triggered from Blynk");
    handleFireDetected();
  }
}
