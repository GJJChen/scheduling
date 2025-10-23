"""
Traffic generation models for network scheduling simulation
"""
import numpy as np
from src.packet import Packet

class TrafficModel:
    """Base class for traffic models"""
    
    def __init__(self, user_id, service_type):
        self.user_id = user_id
        self.service_type = service_type
        self.last_generation_time = 0
        
    def generate_packets(self, current_time, time_step):
        """Generate packets for the current time step"""
        raise NotImplementedError


class PeriodicTrafficModel(TrafficModel):
    """Periodic traffic model - generates packets at fixed intervals"""
    
    def __init__(self, user_id, service_type, packet_size, interval):
        """
        Initialize periodic traffic model
        
        Args:
            user_id: User ID
            service_type: Service type (VO, VI, BE)
            packet_size: Fixed packet size in bytes
            interval: Fixed interval between packets in seconds
        """
        super().__init__(user_id, service_type)
        self.packet_size = packet_size
        self.interval = interval
        self.next_packet_time = interval
        
    def generate_packets(self, current_time, time_step):
        """Generate packets at fixed intervals"""
        packets = []
        
        while current_time >= self.next_packet_time:
            packet = Packet(
                user_id=self.user_id,
                service_type=self.service_type,
                size=self.packet_size,
                arrival_time=self.next_packet_time
            )
            packets.append(packet)
            self.next_packet_time += self.interval
            
        return packets


class PoissonTrafficModel(TrafficModel):
    """Poisson traffic model - generates packets following Poisson distribution"""
    
    def __init__(self, user_id, service_type, packet_size_range, lambda_rate):
        """
        Initialize Poisson traffic model
        
        Args:
            user_id: User ID
            service_type: Service type (VO, VI, BE)
            packet_size_range: Tuple of (min_size, max_size) in bytes
            lambda_rate: Average number of packets per second
        """
        super().__init__(user_id, service_type)
        self.packet_size_range = packet_size_range
        self.lambda_rate = lambda_rate
        
    def generate_packets(self, current_time, time_step):
        """Generate packets following Poisson distribution"""
        packets = []
        
        # Expected number of packets in this time step
        expected_packets = self.lambda_rate * time_step
        
        # Number of packets follows Poisson distribution
        num_packets = np.random.poisson(expected_packets)
        
        for _ in range(num_packets):
            # Random packet size within range
            packet_size = np.random.randint(
                self.packet_size_range[0],
                self.packet_size_range[1] + 1
            )
            
            # Random arrival time within current time step
            arrival_time = current_time - time_step + np.random.random() * time_step
            
            packet = Packet(
                user_id=self.user_id,
                service_type=self.service_type,
                size=packet_size,
                arrival_time=arrival_time
            )
            packets.append(packet)
            
        return packets
