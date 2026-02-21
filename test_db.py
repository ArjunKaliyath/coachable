from cortex import CortexClient

with CortexClient("localhost:50051") as client:
    version, uptime = client.health_check()
    print("Connected to:", version)
    print("Uptime:", uptime)