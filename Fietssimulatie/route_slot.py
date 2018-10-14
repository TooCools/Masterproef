class Slot:

    def __init__(self, t_dc_from, t_dc_target, begin, end):
        self.t_dc_from = t_dc_from
        self.t_dc_target = t_dc_target
        self.begin = begin
        self.end = end

    def is_in_timeslot(self, time):
        return self.begin <= time <= self.end

    def get_change_t_dc(self):
        return (self.t_dc_target - self.t_dc_from) / (self.end - self.begin)


    def is_relevant(self, time):
        return time < self.end
