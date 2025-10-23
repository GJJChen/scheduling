"""
User and Service classes for network scheduling simulation
"""
from collections import deque

class User:
    """Represents a user with multiple services"""
    
    def __init__(self, user_id):
        """
        Initialize a user
        
        Args:
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.services = {}  # service_type -> Service object
        
    def add_service(self, service_type, traffic_model):
        """Add a service to this user"""
        self.services[service_type] = Service(self.user_id, service_type, traffic_model)
        
    def get_service(self, service_type):
        """Get a specific service"""
        return self.services.get(service_type)
    
    def get_all_services(self):
        """Get all services for this user"""
        return list(self.services.values())
    
    def has_service(self, service_type):
        """Check if user has a specific service type"""
        return service_type in self.services


class Service:
    """Represents a service (VO, VI, or BE) for a user"""
    
    def __init__(self, user_id, service_type, traffic_model):
        """
        Initialize a service
        
        Args:
            user_id: ID of the user
            service_type: Type of service (VO, VI, BE)
            traffic_model: TrafficModel instance for generating packets
        """
        self.user_id = user_id
        self.service_type = service_type
        self.traffic_model = traffic_model
        self.queue = deque()
        self.last_transmission_time = 0
        self.total_packets_sent = 0
        self.total_packets_dropped = 0
        self.total_delay = 0
        
    def update_traffic(self, current_time, time_step):
        """Generate new packets and add to queue"""
        new_packets = self.traffic_model.generate_packets(current_time, time_step)
        for packet in new_packets:
            self.queue.append(packet)
            
    def has_packets(self):
        """Check if service has packets waiting"""
        return len(self.queue) > 0
    
    def get_queue_size(self):
        """Get number of packets in queue"""
        return len(self.queue)
    
    def peek_packet(self):
        """Get the first packet without removing it"""
        if self.has_packets():
            return self.queue[0]
        return None
    
    def pop_packet(self):
        """Remove and return the first packet"""
        if self.has_packets():
            return self.queue.popleft()
        return None
    
    def get_max_waiting_time(self, current_time):
        """Get the maximum waiting time among all packets in queue"""
        if not self.has_packets():
            return 0
        return max(packet.get_waiting_time(current_time) for packet in self.queue)
    
    def get_time_since_last_transmission(self, current_time):
        """Get time elapsed since last transmission"""
        return current_time - self.last_transmission_time
    
    def record_transmission(self, current_time, packet):
        """Record a successful transmission"""
        self.last_transmission_time = current_time
        self.total_packets_sent += 1
        waiting_time = packet.get_waiting_time(current_time)
        self.total_delay += waiting_time
        
    def record_drop(self):
        """Record a dropped packet"""
        self.total_packets_dropped += 1
        
    def get_statistics(self):
        """Get service statistics"""
        return {
            'user_id': self.user_id,
            'service_type': self.service_type,
            'packets_sent': self.total_packets_sent,
            'packets_dropped': self.total_packets_dropped,
            'average_delay': self.total_delay / self.total_packets_sent if self.total_packets_sent > 0 else 0
        }
