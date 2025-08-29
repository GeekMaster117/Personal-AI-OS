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
- Nothing runs without you knowing. To stop tracking, simply close the observe.exe window or press Ctrl + C; it stops immediately.
- Your data is stored in data/usagedata.db file. You can choose to delete the file.
- Your data is managed with **SQLite**. You can use SQLite queries to view, update, and delete data from data/usagedata.db file.

## Architecture

Personal AI OS works in three stages, each with modular subsystems:

### Observe

Track how you spend time across your computer.

- **App Monitor** – Tracks your app activity

- **Browser Monitor** – Tracks your web activity (Partially implemented in **App Monitor**)

- **Github Monitor** – Tracks your code activity (Not implemented yet)

- **Input Monitor** – Tracks your peripheral activity (Not implemented yet)

### Reflect

Suggest ways to improve routine, personal and professional life along with productivity insights.

- **Routine Suggestions** – Suggestions on things you do often with a hidden or visible pattern.

- **Professional Suggestions** – Suggestions on improving professional life. (Not implemented yet)

- **Productivity Suggestions** – Suggestions on improving your productivity.

- **Personal Suggestions** – Suggestions on improving your personal life.

### Act (Not implemented yet)

Build automations to improve personal, professional life and improve productivity by automating routines.

- **Suggestion-based automation** - Built by LLM automatically based on the suggestions.

- **Manual automation** - Built by user manually.

## System Diagram

![Personal AI OS Level 1 Level 2](https://github.com/user-attachments/assets/bab6660a-d8b3-452a-8055-88e2a008c7ce)
![Personal AI OS Level 2 Level 3](https://github.com/user-attachments/assets/074fb7f8-7a45-4915-8828-e1928393fdf2)

---

Built by someone who just wanted to understand himself, and got carried away by building an AI OS instead. No cap.

