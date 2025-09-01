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

## How to build the Project (In Windows OS)

### Requirements

- Python 3.10+
- CMake
- Visual Studio 17 2022 with C++ development tools

**Note**: Recommended to do this in a Python virtual environment

### Steps

1. Install the necessary Python packages from dev/requirements.txt
``` shell
pip install -r dev/requirements.txt
```
2. Set up the Visual Studio Environment. Typically found in:
``` shell
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
```
3. Install llama-cpp-python with the following command:
``` shell
pip install llama-cpp-python --global-option=build_ext --global-option="-G Visual Studio 17 2022" --global-option="-DLLAMA_BUILD_SHARED_LIBS" --global-option="-DLLAMA_CUDA=off" --global-option="-DLLAMA_CURL=off" --global-option="-DCMAKE_BUILD_TYPE=Release"
```
4. Go to dev folder and run build.bat to build the project to dist/AI_OS
``` shell
.\build.bat
```

## Architecture

Personal AI OS works in three stages, each with modular subsystems:

### Observe

Track how you spend time across your computer.

- **App Monitor** – Tracks your app activity

- **Browser Monitor** – Tracks your web activity (Partially implemented in **App Monitor**)

- **Github Monitor** – Tracks your code activity (Not implemented yet)

- **Input Monitor** – Tracks your peripheral activity (Not implemented yet)

### Reflect

Suggests ways to improve routine, personal and professional life along with productivity insights.

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
![Personal AI OS Level 3](https://github.com/user-attachments/assets/3e890d22-e69d-4b85-af98-cf6410a83567)

## Core Logic

### Wrappers

#### Llama Wrapper (llama_wrapper.py)

LLaMA wrapper efficiently allocates resources based on availability, and ensures that the model runs within memory limits without causing overflow.

**Step by Step Flow**:
1. Detect Available GPUs:
    - It first checks for the presence of Nvidia GPUs with supported architectures: 61, 75, 86, 89, 120
    - If multiple GPUs exist, it evaluates the best GPU based on free memory. If multiple GPUs have the same free memory, the tie is broken by checking the total memory.
2. Select Optimal GPU Configuration:
    - Once the best GPU is found, it calculates the optimal batch size and GPU layers and tries not to exceed 80% VRAM usage.
    - Latency penalty is computed to balance batch size with the number of layers that can be loaded into VRAM:
     ``` shell
     latency_penalty = (total_layers - layers_loaded) * layer_batchsize_weight
     score = batch_size - latency_penalty
     ```
    - If the optimal batch size is 0, then it will skip GPU acceleration.
3. Fallback to CPU:
    - Falls back to CPU if no compatible GPU exists, or if VRAM is too low for the best GPU.
    - It will try to fit the entire model into RAM.
    - If RAM is insufficient to hold the model, it will raise an error and stop the initialisation.
     ``` shell
     Not enough memory to load the model. Please try closing other applications.
     ```
4. Model Initialisation:
    - After selecting the best device, it will load the model onto it.

**Features**:
- Caching and Loading system prompt
- Chat with the model
- Run inference

#### SQLite Wrapper (sqlite_wrapper.py)

SQLite wrapper provides a thread-safe interface to manage all database operations.

**Step by Step Flow**:
1. Enable Write Ahead Logging.
2. Set Synchronous to Normal.
3. Turn on Foreign keys.

**Features**:
- Execute a single SQL query
- Execute multiple SQL queries in a batch
- Execute a SQL script file.
- Execute multiple queries atomically; either all succeed or all fail.
- Fetch results from a single SQL query
- Fetch results from multiple SQL queries.
- Fetch results from a SQL script file.

---

### Services

#### Suggestion Engine Service (suggestion_engine_service.py)

Suggestion Engine Service manages Llama Wrapper and Cache.

**Step by Step Flow**:
1. Check device configuration:
    - Check if device_config.json exists
    - Validates if the file contains compute optimal batch size for CPU and GPU.
    - If missing or incomplete, it will raise an error to run the benchmark first.
2. Initialise Llama Wrapper with compute optimal batch size for CPU and GPU.

**Features**:
- Option to gracefully close and unload the model.
- Chat with the model.

#### UsageData Service (usagedata_service.py)

Provides an interface for storing, retrieving, and maintaining usage data in a database through the SQLite Wrapper.

**Step by Step Flow**:
1. Connect to SQLite Wrapper.

**Features**:
- Create Schema if not exist. Schema is in sql/schema.sql
- Day Log Operations:
  - Add a new day log
  - Fetch day logs
  - Count available day logs
  - Remove the oldest day log
  - Update the latest day log
- App Log and Title Log Operations:
  - Fetch app logs and title logs
  - Check if specific app names or title names already exist in the database
  - Upsert the latest app log and title log
- App Focus Period, Title Focus Period and Downtime Period Operations:
  - Fetch app focus periods
  - Fetch title focus periods
  - Fetch downtime periods
  - Upsert the latest app focus periods, title focus periods and downtime periods.
 
### Sub Systems

#### Suggestion Engine (suggestion_engine.py)

Suggestion Engine preprocesses data in the Usagedata DB and generates suggestions.

**Step by Step Flow**:
1. Connect to Suggestion Engine Service.
2. Connect to Usagedata DB:
    - if no data is available in Usagedata DB, throws an error:
    ``` shell
    No day logs found in the database.
    ```
3. Preprocess data in Usagedata DB with multithreading:
    - Current day data is processed in detail.
    - Previous days’ data is processed in a condensed format.
  
**Features**:
- Generate Suggestions

#### Usagedata DB (usagedata_db.py)

Usagedata DB manages the storage and integrity of usage data.

**Step by Step Flow**:
1. Check database File:
    - Verifies if the database file exists.
    - If not, create a new database file.
2. Connect to UsageData Service.
3. Create schema if not exist.
4. Ensure Log Integrity:
    - Checks if the latest day log matches the current date.
    - If not, create a new log for the current date.
    - Verifies if the number of logs exceeds the configured limit.
    - If exceeded, it deletes the oldest logs until within the limit.
  
**Features**:
- Update Apps Titles:
  - Check if the drift between monotonic anchor and time anchor is too high, then increment the total anomalies counter and re-anchor both.
  - Check if the difference between current monotonic and last updated monotonic is too large, then log downtime for that period.
  - Update focus period for active app/title and total duration for all open apps/titles.
  - If switched apps/titles, then increment focus count for the active app/title.
  - Set last updated monotonic to current monotonic.
- Get day log IDs.
- Get day log.
- Get app/title log.
- Get app/title focus log.

### Applications

#### Reflect (reflect.py):

Get different types of suggestions, based on usage data.

**Step by Step Flow**:
1. Check if installation has been completed, throw an error if not:
``` shell
Installation verification failed: <reason>
Please run install.exe
```
2. Connect to Suggestion Engine:
    - Request it to preprocess logs.
3. Open Menu:
    - Show available suggestion types (Routine, Personal, Productivity).
4. Generate Suggestion:
    - Send a signal to Suggestion Engine to generate the requested suggestion.
  
#### Observe (observe.py):

Monitor usage data to get suggestions.

**Step by Step Flow**:
1. Connect to Usagedata DB.
2. Fetch currently open apps/titles and the active app/title.
3. Upsert data to Usagedata DB.
4. Sleep briefly, and repeat until a shutdown signal is received.

---

Built by someone who just wanted to understand himself, and got carried away by building an AI OS instead. No cap.

