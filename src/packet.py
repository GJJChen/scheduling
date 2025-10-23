"""
Packet class for network scheduling simulation
"""
import time

class Packet:
    """Represents a network packet"""
    
    def __init__(self, user_id, service_type, size, arrival_time):
        """
        Initialize a packet
        
        Args:
            user_id: ID of the user who generated the packet
            service_type: Type of service (VO, VI, BE)
            size: Packet size in bytes
            arrival_time: Time when packet arrived (in seconds)
        """
        self.user_id = user_id
        self.service_type = service_type
        self.size = size
        self.arrival_time = arrival_time
        self.transmission_time = None
        
    def get_waiting_time(self, current_time):
        """Calculate how long the packet has been waiting"""
        return current_time - self.arrival_time
    
    def __repr__(self):
        return f"Packet(user={self.user_id}, service={self.service_type}, size={self.size}B, arrival={self.arrival_time:.6f})"
