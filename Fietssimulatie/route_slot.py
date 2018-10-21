class Slot:

    def __init__(self, t_dc_from, t_dc_target, slope_from, slope_target, rpm_change, begin, end):
        self.t_dc_from = t_dc_from
        self.t_dc_target = t_dc_target
        self.slope_from = slope_from
        self.slope_target = slope_target
        self.begin = begin
        self.end = end
        self.rpm_change = rpm_change

    def is_in_timeslot(self, time):
        return self.begin <= time <= self.end

    def get_change_t_dc(self):
        return (self.t_dc_target - self.t_dc_from) / (self.end - self.begin)

    def get_change_slope(self):
        return (self.slope_target - self.slope_from) / (self.end - self.begin)

    def is_relevant(self, time):
        return time < self.end
