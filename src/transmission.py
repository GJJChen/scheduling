"""
Transmission time calculator for network scheduling simulation
"""
import config

class TransmissionCalculator:
    """Calculate transmission time for packets"""
    
    def __init__(self):
        self.overhead = config.EDCA_PREAMBLE_OVERHEAD
        self.rate = config.MCS11_RATE
        self.max_time = config.MAX_TRANSMISSION_TIME
        
    def calculate_transmission_time(self, packet_size_bytes):
        """
        Calculate transmission time for a packet
        
        Args:
            packet_size_bytes: Packet size in bytes
            
        Returns:
            Transmission time in seconds
        """
        # Convert bytes to bits
        packet_size_bits = packet_size_bytes * 8
        
        # Calculate transmission time: overhead + data transmission time
        transmission_time = self.overhead + (packet_size_bits / self.rate)
        
        # Cap at maximum transmission time
        transmission_time = min(transmission_time, self.max_time)
        
        return transmission_time
    
    def calculate_throughput(self, packet_size_bytes, transmission_time):
        """
        Calculate effective throughput
        
        Args:
            packet_size_bytes: Packet size in bytes
            transmission_time: Actual transmission time in seconds
            
        Returns:
            Throughput in Mbps
        """
        packet_size_bits = packet_size_bytes * 8
        throughput_bps = packet_size_bits / transmission_time
        throughput_mbps = throughput_bps / 1e6
        return throughput_mbps
