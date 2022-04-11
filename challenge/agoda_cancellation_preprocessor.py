import pandas as pd


class AgodaCancellationPreprocessor:
    SATURDAY = 5

    def __init__(self, full_data: pd.DataFrame):
        self.number_of_times_customer_canceled = dict()
        for id, cancellation in full_data[
            ["h_customer_id", "cancellation_datetime"]].itertuples(
            index=False):
            if cancellation == 0:
                if id in self.number_of_times_customer_canceled:
                    self.number_of_times_customer_canceled[id] += 1
                else:
                    self.number_of_times_customer_canceled[id] = 1
        self.average_cancellation_days_from_booking = dict()
        self.average_cancellation_days_to_checkin = dict()
        dates = pd.DataFrame([])
        dates["cancellation_datetime"] = pd.to_datetime(full_data["cancellation_datetime"])
        dates["booking_datetime"] = pd.to_datetime(full_data["booking_datetime"])
        dates["checkin_date"] = pd.to_datetime(full_data["checkin_date"])
        for id, cancellation, booking_date, checkin_date in pd.concat([full_data[
            "h_customer_id"], dates["cancellation_datetime"], dates["booking_datetime"], dates["checkin_date"]], axis=1).itertuples(
            index=False):
            if cancellation == 0:
                if id in self.average_cancellation_days_from_booking:
                    self.average_cancellation_days_from_booking[id] += (cancellation - booking_date).days/self.number_of_times_customer_canceled[id]
                    self.average_cancellation_days_to_checkin[id] += (checkin_date - cancellation).days/self.number_of_times_customer_canceled[id]
                else:
                    self.average_cancellation_days_from_booking[id] = (cancellation - booking_date).days/self.number_of_times_customer_canceled[id]
                    self.average_cancellation_days_to_checkin[id] = (checkin_date - cancellation).days /self.number_of_times_customer_canceled[id]

    def preprocess(self, full_data: pd.DataFrame) -> pd.DataFrame:
        # take features that need no processing.
        features = full_data[["h_booking_id",
                              "hotel_star_rating",
                              "guest_is_not_the_customer",
                              "no_of_adults",
                              "no_of_children",
                              "no_of_extra_bed",
                              "no_of_room",
                              "original_selling_amount",
                              "is_user_logged_in",
                              "is_first_booking",
                              "request_nonesmoke",
                              "request_latecheckin",
                              "request_highfloor",
                              "request_largebed",
                              "request_twinbeds",
                              "request_airport",
                              "request_earlycheckin"]].copy()

        # add time related features
        booking_date = pd.to_datetime(full_data["booking_datetime"])
        checkin_date = pd.to_datetime(full_data["checkin_date"])
        checkout_date = pd.to_datetime(full_data["checkout_date"])
        features["hotel_live_time"] = (pd.Timestamp.now() - pd.to_datetime(
            full_data.hotel_live_date)).dt.days
        features["booking_checkin_difference"] = (
                checkin_date - booking_date).dt.days
        features["length_of_stay"] = (checkout_date - checkin_date).dt.days
        arrival_day = checkin_date.apply(lambda x: x.weekday())
        features["stay_over_weekend"] = (features["length_of_stay"] > 6) | (
                (arrival_day <= AgodaCancellationPreprocessor.SATURDAY) & (
                AgodaCancellationPreprocessor.SATURDAY <= (
                arrival_day + features[
            "length_of_stay"])))

        # add dummies
        features = pd.concat([features,
                              pd.get_dummies(full_data.accommadation_type_name,
                                             drop_first=True),
                              pd.get_dummies(full_data.charge_option,
                                             drop_first=True)], axis=1)

        # add cancellation history feature
        features["has_cancellation_history"] = full_data[
            "h_customer_id"].apply(
            self._number_of_times_cancelled)

        features["average_cancellation_days_from_booking"] = full_data[
            "h_customer_id"].apply(
            self._get_average_cancellation_days_from_booking)

        features["average_cancellation_days_to_checkin"] = full_data[
            "h_customer_id"].apply(
            self._get_average_cancellation_days_to_checkin)

        # add cancellation policy related features
        features["cancellation_policy_at_time_of_order"] = pd.concat(
            [features[["booking_checkin_difference", "length_of_stay"]],
             full_data["cancellation_policy_code"]], axis=1).apply(
            lambda x: AgodaCancellationPreprocessor._current_policy(
                x["booking_checkin_difference"],
                x["length_of_stay"],
                x["cancellation_policy_code"]), axis=1)
        cancellation_window_start_diff = features.booking_checkin_difference - 7
        cancellation_window_start_diff.name = "cancellation_window_start"
        features[
            "cancellation_policy_at_start_of_cancellation_window"] = pd.concat(
            [cancellation_window_start_diff, features["length_of_stay"],
             full_data["cancellation_policy_code"]], axis=1).apply(
            lambda x: AgodaCancellationPreprocessor._current_policy(
                x["cancellation_window_start"],
                x["length_of_stay"],
                x["cancellation_policy_code"]), axis=1)
        cancellation_window_end_diff = features.booking_checkin_difference - 35
        cancellation_window_end_diff.name = "cancellation_window_end"
        features[
            "cancellation_policy_at_end_of_cancellation_window"] = pd.concat(
            [cancellation_window_end_diff, features["length_of_stay"],
             full_data["cancellation_policy_code"]], axis=1).apply(
            lambda x: AgodaCancellationPreprocessor._current_policy(
                x["cancellation_window_end"],
                x["length_of_stay"],
                x["cancellation_policy_code"]), axis=1)
        features["cancellation_policy_change_during_window"] = \
            features.cancellation_policy_at_end_of_cancellation_window - \
            features.cancellation_policy_at_start_of_cancellation_window
        return features.fillna(0)

    @staticmethod
    def _current_policy(days_from_checkin: pd.Series,
                        length_of_stay: pd.Series,
                        penalty_code: pd.Series) -> pd.Series:
        penalties = []
        for penalty in penalty_code.split("_"):
            if "D" not in penalty:
                continue
            penalty_days, penalty_calculation = penalty.split("D")
            if penalty_calculation[-1] == "N":
                percentage = int(penalty_calculation[:-1]) / length_of_stay
            else:
                percentage = float(penalty_calculation[:-1])
            penalties.append((float(penalty_days), percentage))
        penalties.sort(key=lambda x: x[0], reverse=True)
        current_penalty = 0
        for days, penalty in penalties:
            if days < days_from_checkin:
                break
            current_penalty = penalty
        return current_penalty

    def preprocess_labels(self, cancellation_date: pd.Series,
                          booking_datetime: pd.Series) -> pd.Series:
        cancellation = cancellation_date.apply(
            lambda x: pd.Timestamp.now() if x == 0 else pd.to_datetime(x))
        booking = pd.to_datetime(booking_datetime)
        diff = (pd.to_datetime(cancellation, unit="s") - pd.to_datetime(
            booking,
            unit="s")).dt.days
        return (diff >= 7) & (diff < 35)

    def _number_of_times_cancelled(self, id: int) -> int:
        if id in self.number_of_times_customer_canceled:
            return self.number_of_times_customer_canceled[id]
        return 0

    def _get_average_cancellation_days_from_booking(self, id: int) -> int:
        if id in self.average_cancellation_days_from_booking:
            return self.average_cancellation_days_from_booking[id]
        return 1000

    def _get_average_cancellation_days_to_checkin(self, id: int) -> int:
        if id in self.average_cancellation_days_to_checkin:
            return self.average_cancellation_days_to_checkin[id]
        return -1000
