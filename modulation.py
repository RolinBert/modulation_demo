from numpy import arange, sin, cos, pi, zeros_like, reshape, array, packbits


class DigitalModulation:
    def __init__(self, carrier_freq_hz=1e6, demo_duration_s=8e-6) -> None:
        # Carrier frequency in Hz
        self.carrier_freq_hz = 1e6
        # Demo duration in seconds
        self.demo_duration_s = 8e-6

        # Oversampling frequency of the generated demo values compared to the carrier frequeny
        self._oversampling = int(32)
        # Sampling frequency of this demo in hertz
        self._sample_freq_hz = carrier_freq_hz * self._oversampling
        # Sampling period of this demo in seconds
        self._sample_period_s = 1 / self._sample_freq_hz
        # Start phase in radian
        self.start_phase_rad = 0

        # Sampling points in seconds
        self.sample_points_s = arange(
            0, demo_duration_s, self._sample_period_s)

    def get_carrier(self):
        # Sample values of the carrier in Volt
        carrier_values_v = sin(
            2 * pi * self.carrier_freq_hz * self.sample_points_s)
        return carrier_values_v

    def get_ask_mod(self, data_vec, bits_per_symbol_log2=1):
        """Generate a amplitude modulated signal
           If the signal contains multiple bits per symbol, the amplitude mapping is done
           in little endian order => (e.g. 1011 => 2**0 + 2**2 + 2**3 = 13)

        Parameters
        ----------
        data_vec : array_like
            Binary data values that should be modulated (list or array)
        bits_per_symbol_log2 : int
            Log2 of the number of bits per symbol. Maximum supported value = 8
            (e.g. 1 = 2 bits per symbol, 2 => 4 bits per symbol, ...)

        Returns
        -------
        ndarray
            Modulated carrier values
        """
        # Calculate how many carrier periods on bit spans
        periods_per_bit = (self.carrier_freq_hz *
                           self.demo_duration_s) // len(data_vec)
        # Calculate the same value but this time with respects to how many samples
        mod_samples_per_bit = int(periods_per_bit * self._oversampling)
        # Sample values of the carrier in Volt
        modulated_values_v = zeros_like(self.sample_points_s)

        # Helper variable holding the angular frequency ω=2πf
        angular_frequency_hz = 2 * pi * self.carrier_freq_hz
        # Reshape the data into pairs (0,1,1,1) => ((0,1), (1,1))
        data_vec_bool = array(data_vec).astype(bool)
        ask_data_vec = reshape(data_vec_bool, (-1, bits_per_symbol_log2))
        amplitude_divider = 2**bits_per_symbol_log2

        for idx, data in enumerate(ask_data_vec):
            modulated_value = packbits(data, bitorder='little')[0]
            amp_value = (modulated_value + 1) / amplitude_divider
            start_sample_idx = idx * self._oversampling
            stop_sample_idx = start_sample_idx + mod_samples_per_bit
            modulated_values_v[start_sample_idx:stop_sample_idx] =\
                amp_value * sin(angular_frequency_hz *
                                self.sample_points_s[start_sample_idx:stop_sample_idx] + self.start_phase_rad)

        return modulated_values_v

    def get_psk_mod(self, data_vec, bits_per_symbol_log2=1):
        """Generate a phase modulated signal
           If the signal contains multiple bits per symbol, the frequency mapping is done
           in little endian order => (e.g. 1011 => 2**0 + 2**2 + 2**3 = 13).
           In the example the resulting frequency would be 13/16 * max_freq

        Parameters
        ----------
        data_vec : array_like
            Binary data values that should be modulated (list or array)
        bits_per_symbol_log2 : int
            Log2 of the number of bits per symbol. Maximum supported value = 8
            (e.g. 1 = 2 bits per symbol, 2 => 4 bits per symbol, ...)

        Returns
        -------
        ndarray
            Modulated carrier values
        """
        # Calculate how many carrier periods on bit spans
        periods_per_bit = (self.carrier_freq_hz *
                           self.demo_duration_s) // len(data_vec)
        # Calculate the same value but this time with respects to how many samples
        mod_samples_per_bit = int(periods_per_bit * self._oversampling)

        # Sample values of the carrier in Volt
        modulated_values_v = zeros_like(self.sample_points_s)

        # Helper variable holding the angular frequency ω=2πf
        angular_frequency_hz = 2 * pi * self.carrier_freq_hz

        # Reshape the data into pairs (0,1,1,1) => ((0,1), (1,1))
        data_vec_bool = array(data_vec).astype(bool)
        ask_data_vec = reshape(data_vec_bool, (-1, bits_per_symbol_log2))
        phase_divider = 2**bits_per_symbol_log2

        for idx, data in enumerate(ask_data_vec):

            modulated_value = (packbits(data, bitorder='little')[0] + 1)
            phase_value_rad = self.start_phase_rad + \
                2 * pi * modulated_value / phase_divider
            start_sample_idx = idx * self._oversampling
            stop_sample_idx = start_sample_idx + mod_samples_per_bit
            modulated_values_v[start_sample_idx:stop_sample_idx] =\
                sin(angular_frequency_hz *
                    self.sample_points_s[start_sample_idx:stop_sample_idx] + phase_value_rad)

        return modulated_values_v

    def get_fsk_mod(self, data_vec, frequency_span, bits_per_symbol_log2=1):
        """Generate a frequency modulated signal
           If the signal contains multiple bits per symbol, the frequency mapping is done
           in little endian order => (e.g. 1011 => 2**0 + 2**2 + 2**3 = 13).
           In the example the resulting frequency would be 13/16 * max_freq

        Parameters
        ----------
        data_vec : array_like
            Binary data values that should be modulated (list or array)
        bits_per_symbol_log2 : int
            Log2 of the number of bits per symbol. Maximum supported value = 8
            (e.g. 1 = 2 bits per symbol, 2 => 4 bits per symbol, ...)

        Returns
        -------
        ndarray
            Modulated carrier values
        """
        # Calculate how many carrier periods on bit spans
        periods_per_bit = (self.carrier_freq_hz *
                           self.demo_duration_s) // len(data_vec)
        # Calculate the same value but this time with respects to how many samples
        mod_samples_per_bit = int(periods_per_bit * self._oversampling)

        # FSK Modulation index η=Δf*T, Δf:Frequency span, T:Symbol duration
        # carrier_period_s = 1 / self.carrier_freq_hz
        # symbol_duration_s = periods_per_bit * carrier_period_s
        # mod_index = frequency_span * symbol_duration_s
        frequency_min = self.carrier_freq_hz - frequency_span / 2
        # Sample values of the carrier in Volt
        modulated_values_v = zeros_like(self.sample_points_s)

        # Reshape the data into pairs (0,1,1,1) => ((0,1), (1,1))
        data_vec_bool = array(data_vec).astype(bool)
        fsk_data_vec = reshape(data_vec_bool, (-1, bits_per_symbol_log2))
        freq_span_divider = 2**bits_per_symbol_log2-1

        for idx, data in enumerate(fsk_data_vec):
            modulated_value = packbits(data, bitorder='little')[0]
            freq_value_hz = frequency_min + modulated_value * \
                frequency_span / freq_span_divider
            start_sample_idx = idx * self._oversampling
            stop_sample_idx = start_sample_idx + mod_samples_per_bit
            # Helper variable holding the angular frequency ω=2πf
            angular_frequency_hz = 2 * pi * freq_value_hz
            modulated_values_v[start_sample_idx:stop_sample_idx] =\
                sin(angular_frequency_hz *
                    self.sample_points_s[start_sample_idx:stop_sample_idx] + self.start_phase_rad)

        return modulated_values_v

    def get_qam_mod(self, data_vec, bits_per_symbol_log2=2):
        """Generate a N-QAM modulated signal

        Parameters
        ----------
        data_vec : array_like
            Binary data values that should be modulated (list or array)
        bits_per_symbol_log2 : int
            Log2 of the number of bits per symbol. Value must be in the range [2, 8]
            (e.g. 1 = 2 bits per symbol, 2 => 4 bits per symbol, ...)

        Returns
        -------
        ndarray
            Modulated carrier values
        """
        # Calculate how many carrier periods on bit spans
        periods_per_bit = (self.carrier_freq_hz *
                           self.demo_duration_s) // len(data_vec)
        # Calculate the same value but this time with respects to how many samples
        mod_samples_per_bit = int(periods_per_bit * self._oversampling)
        # Sample values of the carrier in Volt
        modulated_values_v = zeros_like(self.sample_points_s, dtype=complex)

        # Helper variable holding the angular frequency ω=2πf
        angular_frequency_hz = 2 * pi * self.carrier_freq_hz
        # Reshape the data into pairs (0,1,1,1) => ((0,1), (1,1))
        data_vec_bool = array(data_vec).astype(bool)
        data_vec_qam = reshape(data_vec_bool, (-1, 2, bits_per_symbol_log2//2))
        qam_idx = 2**bits_per_symbol_log2
        for idx, data in enumerate(data_vec_qam):
            modulated_value = (packbits(data, bitorder='little')[0] + 1)
            i_val, q_val = self.get_qam_modulation_mapping(
                modulated_value, qam_idx)
            start_sample_idx = idx * self._oversampling
            stop_sample_idx = start_sample_idx + mod_samples_per_bit
            modulated_values_v[start_sample_idx:stop_sample_idx] =\
                (i_val * cos(angular_frequency_hz * self.sample_points_s[start_sample_idx:stop_sample_idx])
                 + 1j * q_val * sin(angular_frequency_hz * self.sample_points_s[start_sample_idx:stop_sample_idx]))

        return modulated_values_v

    def get_qam_modulation_mapping(self, value, qam_idx):
        # Return is (I, Q)
        ret_val = array((0, 0))
        if qam_idx == 4:
            if value == 0b00:
                ret_val = array((1, 1))
            elif value == 0b01:
                ret_val = array((1, -1))
            elif value == 0b10:
                ret_val = array((-1, 1))
            elif value == 0b11:
                ret_val = array((-1, -1))
        elif qam_idx == 8:
            if value == 0b000:
                ret_val = array((3, 0))
            elif value == 0b001:
                ret_val = array((1, 1))
            elif value == 0b010:
                ret_val = array((-1, 1))
            elif value == 0b011:
                ret_val = array((0, 3))
            elif value == 0b100:
                ret_val = array((1, -1))
            elif value == 0b101:
                ret_val = array((0, -3))
            elif value == 0b110:
                ret_val = array((-3, 0))
            elif value == 0b111:
                ret_val = array((-1, -1))
        elif qam_idx == 16:
            if value == 0b0000:
                ret_val = array((1, 1))
            elif value == 0b0001:
                ret_val = array((3, 1))
            elif value == 0b0010:
                ret_val = array((1, 3))
            elif value == 0b0011:
                ret_val = array((3, 3))
            elif value == 0b0100:
                ret_val = array((1, -1))
            elif value == 0b0101:
                ret_val = array((3, -1))
            elif value == 0b0110:
                ret_val = array((1, -3))
            elif value == 0b0111:
                ret_val = array((3, -3))
            elif value == 0b1000:
                ret_val = array((-1, 1))
            elif value == 0b1001:
                ret_val = array((-3, 1))
            elif value == 0b1010:
                ret_val = array((-1, 3))
            elif value == 0b1011:
                ret_val = array((-3, 3))
            elif value == 0b1100:
                ret_val = array((-1, -1))
            elif value == 0b1101:
                ret_val = array((-3, -1))
            elif value == 0b1110:
                ret_val = array((-1, -3))
            elif value == 0b1111:
                ret_val = array((-3, -3))

        return ret_val
