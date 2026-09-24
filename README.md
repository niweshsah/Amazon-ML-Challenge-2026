# Amazon-ML-Challenge-2026

## Docker Environment

Use the following steps to access the Docker environment.

### 1. SSH into the Server

Connect to the server while on IIT Wi-Fi:

```bash
ssh user@172.18.18.32
```

Enter the password provided to you.

### 2. Go to the Workspace

```bash
cd ~/workspace/niwesh
```

### 3. Start/Enter the Docker Environment

Run:

```bash
./run_qwen.sh
```

This will start the Docker container and open a terminal inside it.

If the container is already running, running `./run_qwen.sh` will open a new terminal connected to the existing container.

## Installing Packages

Do **not** install packages directly using `pip install`.

Any changes made to the Docker environment are lost when the container is exited.

If you need to install a new package or modify the environment, contact me.
