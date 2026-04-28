import jax

print("Available JAX devices:")
for i, device in enumerate(jax.devices()):
    print(f"  Device {i}: {device.platform.upper()} - {device.device_kind}")
