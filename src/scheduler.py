"""
Traditional scheduling algorithm implementation
"""
import config

class TraditionalScheduler:
    """
    Traditional scheduling algorithm with priority-based rules
    
    Rules:
    1. VO packets: Strict priority, send longest waiting first, drop if delay > 20ms
    2. VI packets: Priority if delay > 20ms, drop if > 50ms, otherwise continue
    3. BE packets: Check bandwidth requirement (100ms minimum interval)
    4. Fallback: Send VI/BE packet with longest size
    """
    
    def __init__(self):
        self.vo_threshold = config.VO_DELAY_THRESHOLD
        self.vi_priority_threshold = config.VI_DELAY_THRESHOLD_PRIORITY
        self.vi_drop_threshold = config.VI_DELAY_THRESHOLD_DROP
        self.be_min_interval = config.BE_MIN_INTERVAL
        self.negative_events = []
        
    def schedule(self, services, current_time):
        """
        Select the next packet to transmit based on traditional scheduling rules
        
        Args:
            services: List of Service objects
            current_time: Current simulation time
            
        Returns:
            Tuple of (selected_service, selected_packet) or (None, None) if no packet to send
        """
        # First, check and drop packets that exceed thresholds
        self._check_and_drop_packets(services, current_time)
        
        # Rule 1: Check for VO packets
        vo_services = [s for s in services if s.service_type == 'VO' and s.has_packets()]
        if vo_services:
            return self._select_vo_packet(vo_services, current_time)
        
        # Rule 2: Check for VI packets with delay > 20ms
        vi_services = [s for s in services if s.service_type == 'VI' and s.has_packets()]
        priority_vi = [s for s in vi_services 
                       if s.get_max_waiting_time(current_time) > self.vi_priority_threshold]
        if priority_vi:
            return self._select_priority_vi_packet(priority_vi, current_time)
        
        # Rule 3: Check BE bandwidth requirements
        be_services = [s for s in services if s.service_type == 'BE' and s.has_packets()]
        be_need_service = [s for s in be_services 
                          if s.get_time_since_last_transmission(current_time) >= self.be_min_interval]
        if be_need_service:
            return self._select_be_packet(be_need_service, current_time)
        
        # Rule 4: Fallback - select VI/BE packet with longest size
        vi_be_services = vi_services + be_services
        if vi_be_services:
            return self._select_longest_packet(vi_be_services)
        
        return None, None
    
    def _check_and_drop_packets(self, services, current_time):
        """Check and drop packets that exceed delay thresholds"""
        for service in services:
            if not service.has_packets():
                continue
                
            packets_to_drop = []
            
            if service.service_type == 'VO':
                # Drop VO packets with delay > 20ms
                for packet in service.queue:
                    if packet.get_waiting_time(current_time) > self.vo_threshold:
                        packets_to_drop.append(packet)
                        
            elif service.service_type == 'VI':
                # Drop VI packets with delay > 50ms
                for packet in service.queue:
                    if packet.get_waiting_time(current_time) > self.vi_drop_threshold:
                        packets_to_drop.append(packet)
            
            # Remove dropped packets and record negative events
            for packet in packets_to_drop:
                service.queue.remove(packet)
                service.record_drop()
                self.negative_events.append({
                    'time': current_time,
                    'user_id': service.user_id,
                    'service_type': service.service_type,
                    'delay': packet.get_waiting_time(current_time),
                    'reason': 'delay_exceeded'
                })
    
    def _select_vo_packet(self, vo_services, current_time):
        """Select VO packet with longest waiting time"""
        selected_service = max(vo_services, 
                              key=lambda s: s.get_max_waiting_time(current_time))
        selected_packet = selected_service.pop_packet()
        return selected_service, selected_packet
    
    def _select_priority_vi_packet(self, priority_vi_services, current_time):
        """Select priority VI packet with longest waiting time"""
        selected_service = max(priority_vi_services,
                              key=lambda s: s.get_max_waiting_time(current_time))
        selected_packet = selected_service.pop_packet()
        return selected_service, selected_packet
    
    def _select_be_packet(self, be_services, current_time):
        """Select BE packet with longest waiting time among those needing service"""
        selected_service = max(be_services,
                              key=lambda s: s.get_max_waiting_time(current_time))
        selected_packet = selected_service.pop_packet()
        return selected_service, selected_packet
    
    def _select_longest_packet(self, services):
        """Select packet with longest size from VI/BE services"""
        selected_service = None
        selected_packet = None
        max_size = 0
        
        for service in services:
            packet = service.peek_packet()
            if packet and packet.size > max_size:
                max_size = packet.size
                selected_service = service
        
        if selected_service:
            selected_packet = selected_service.pop_packet()
            
        return selected_service, selected_packet
    
    def get_negative_events(self):
        """Get list of negative events (dropped packets)"""
        return self.negative_events
    
    def clear_negative_events(self):
        """Clear negative events list"""
        self.negative_events = []
