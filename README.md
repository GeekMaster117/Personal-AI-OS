# Personal AI OS

An AI Meta Operating System, which monitors your data, gives suggestions based on data, and builds automations based on suggestions.

## Core Philosophy

- Completely Offline, but still intelligent.

- Run on low-end devices.

## How to Use

### Installation

- Run **install.exe**
- This will install the model and run benchmarks on CPU and GPU if available.

### Monitoring

- Run **observe.exe**
- This will monitor your app data.
- Data monitored:
  - Apps you use and their titles. (Used for suggestions)
  - The time you spend actively, passively and total duration on each app and its titles. (Used to find patterns over a span of days)
  - Number of times you switch between apps and their titles. (Used to find distractions)
  - The duration **observe.exe** has been offline. (Model avoids suggestions during this time frame, and to know how long you have been offline)
  - Number of times the system clock has been changed. (If the count is too high, the model knows the data may be invalid).
 
### Suggestions

- Run **reflect.exe**
- Gives suggestions based on the monitored data.
- Suggestions available:
  - **Routine Suggestions**: Describes when and how consistently apps/titles are used, or when idle recovery occurs.
  - **Productivity Suggestions**: Suggests workflow adjustments, like longer uninterrupted blocks or protecting peak focus hours for deep work.
  - **Personal Suggestions**: Suggest small lifestyle adjustments, like taking breaks or avoiding late-night sessions.
 
## Privacy

- It runs locally. Your data is never sent online.
- Your data is stored in data/usagedata.db file. You can choose to delete the file.
- Your data is managed with **SQLite**. You can use SQLite queries to view, update, and delete data from data/usagedata.db file.

## Architecture

Sentinel works in three stages, each with modular subsystems:

### Observe

Track how you spend time across your computer.

- **App Monitor** – Tracks your app activity

- **Browser Monitor** – Tracks your web activity

- **Github Monitor** – Tracks you code activity

- **Input Monitor** – Tracks your peripheral activity

### Reflect

Suggest ways to improve routine, personal and professional life along with productivity insights.

- **Routine Suggestions** – Suggestions on things you do often with a hidden or visible pattern.

- **Professional Suggestions** – Suggestions on improving professional life.

- **Productivity Suggestions** – Suggestions on improving your productivity.

- **Personal Suggestions** – Suggestions on improving your personal life.

### Act

Build automations to improve personal, professional life and improve productivity by automating routines.

- **Suggestion-based automation** - Built by LLM automatically based on the suggestions.

- **Manual automation** - Built by user manually.

## System Diagram

![image](https://github.com/user-attachments/assets/bab6660a-d8b3-452a-8055-88e2a008c7ce)

## Privacy and Ethics

- It runs locally. You own your data.

- You can choose to delete your data at any time.

---

Built by someone who just wanted to understand himself, and got carried away by building an AI OS instead. No cap.

