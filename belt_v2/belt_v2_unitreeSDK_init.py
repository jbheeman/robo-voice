import unitree_sdk2
from unitree_sdk2.go2 import SportClient
import sys

_motion_client = None

def get_robot_client(interface="wlan0"): # Connected to the robot over internet
  global _motion_client
  if _motion_client is not None:
      
    return _motion_client

  try:
        # Initialize the low-level network layer
    unitree_sdk2.ChannelFactory.Initialize(0, interface)
        
  
    client = SportClient("")
    client.SetTimeout(5.0)
    client.Init()
        
    _motion_client = client
    return _motion_client
        
  except Exception as e:
    print(f"[Hardware Error] Critical SDK initialization failure: {e}")
    sys.exit(1)
