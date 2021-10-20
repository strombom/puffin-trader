from Indicators.supersmoother import SuperSmoother


class PredictionFilter:
    def __init__(self):
        self.ss_threshold = SuperSmoother(period=20, initial_value=0.0)
        self.ss_smooth = SuperSmoother(period=200, initial_value=0.25)
        self.threshold = []
        self.smooth = []

    def append(self, value):
        if value > 0:
            self.threshold.append(self.ss_threshold.append(max(0, value)))
        elif len(self.threshold) > 0:
            self.threshold.append(self.threshold[-1])
        else:
            self.threshold.append(0.0)

        if value > self.threshold[-1] * 2:
            self.smooth.append(self.ss_smooth.append(max(0, value) * 1.25))
        elif len(self.smooth) > 0:
            self.smooth.append(self.smooth[-1])
        else:
            self.smooth.append(0)
