import math
import logging
import csv
import json
import pprint # For printing parameters nicely

# --- RateBasedAnalyzer class definition remains the same ---
# (Paste the RateBasedAnalyzer class code here)
class RateBasedAnalyzer(object): # Removed DrowsinessAnalyzer inheritance for standalone testing
    """
    Revised V3: Rate-based analysis with conditional overrides, non-linear damping,
    and averaged eye scores. - ITERATION 2 REVISED PARAMETERS
    """

    def __init__(self,
                 # --- Basic Thresholds ---
                 perclos_threshold=14.0,
                 max_closure_duration_threshold=0.4,
                 yawn_rate_threshold=1.5,

                 # --- Extreme Thresholds for Conditional Overrides ---
                 extreme_perclos_threshold=45.0,
                 extreme_duration_threshold=1.5,
                 extreme_yawn_rate_threshold=12.0,
                 override_max_normal_perc=55.0, # Potential adjustment point

                 # --- Score Calculation Parameters ---
                 perclos_scale=1.2,
                 duration_scale=1.8,
                 yawn_rate_scale=1.0,
                 score_cap=2.5,

                 # --- Weights (Rebalanced - Less Yawn Influence) ---
                 eye_metric_weight=0.7,  # Increased eye weight
                 yawn_metric_weight=0.3,  # Reduced yawn weight

                 # --- Non-Linear Damping Parameters ---
                 damping_base_factor=0.6, # Potential adjustment point
                 damping_power=1.8,      # Potential adjustment point

                 # --- Decision Making ---
                 drowsiness_decision_threshold=0.50, # Potential adjustment point

                 # --- Minimum requirements ---
                 minimum_frames_for_analysis=30,
                 fps=20
                 ):
        """
        Initialize V3 analyzer with conditional overrides and non-linear damping. - ITERATION 2 REVISED PARAMETERS
        """
        # Store all parameters
        self.perclos_threshold = perclos_threshold
        self.max_closure_duration_threshold = max_closure_duration_threshold
        self.yawn_rate_threshold = yawn_rate_threshold
        self.extreme_perclos_threshold = extreme_perclos_threshold
        self.extreme_duration_threshold = extreme_duration_threshold
        self.extreme_yawn_rate_threshold = extreme_yawn_rate_threshold
        self.override_max_normal_perc = override_max_normal_perc
        self.perclos_scale = perclos_scale
        self.duration_scale = duration_scale
        self.yawn_rate_scale = yawn_rate_scale
        self.score_cap = score_cap
        self.eye_metric_weight = eye_metric_weight
        self.yawn_metric_weight = yawn_metric_weight
        self.damping_base_factor = damping_base_factor
        self.damping_power = damping_power
        self.drowsiness_decision_threshold = drowsiness_decision_threshold
        self.minimum_frames_for_analysis = minimum_frames_for_analysis
        self.default_fps = fps

        # Basic validation
        assert 0 <= eye_metric_weight <= 1
        assert 0 <= yawn_metric_weight <= 1
        assert 0 <= self.override_max_normal_perc <= 100
        assert self.damping_power > 0


    def _calculate_metric_score(self, value, threshold, scale):
        """Calculates a score based on how much a value exceeds a threshold, with capping."""
        if value > threshold and threshold > 0:
            score = ((value - threshold) / threshold) * scale
            return min(score, self.score_cap)
        return 0.0

    def _calculate_damping(self, normal_state_percentage):
        """Calculates damping amount using a non-linear function."""
        if normal_state_percentage <= 0:
            return 0.0
        # Damping = base * (normal_perc / 100) ^ power
        damping_fraction = normal_state_percentage / 100.0
        damping_amount = self.damping_base_factor * math.pow(damping_fraction, self.damping_power)
        # Ensure damping doesn't exceed 1.0 (or slightly less to avoid zeroing out score)
        return min(damping_amount, 0.99)


    def analyze(self, detection_results):
        """
        Analyzes detection results using V3 logic: conditional overrides, non-linear damping.
        """
        # --- 1. Extract Data & Basic Checks ---
        yawn_count = detection_results.get('yawn_count', 0)
        eye_closed_detection_count = detection_results.get('eye_closed_frames', 0) # Might be less reliable than total_eye_closed_frames
        total_eye_closed_frames = detection_results.get('total_eye_closed_frames', 0)
        max_consecutive_eye_closed = detection_results.get('max_consecutive_eye_closed', 0)
        normal_state_frames = detection_results.get('normal_state_frames', 0)
        total_frames = detection_results.get('total_frames', 0)
        fps = detection_results.get('fps', self.default_fps) # FPS from details

        if fps <= 0:
            logging.warning(f"Invalid FPS ({fps}). Using default FPS: {self.default_fps}")
            fps = self.default_fps

        if total_frames < self.minimum_frames_for_analysis:
            logging.info(f"Insufficient frames ({total_frames} < {self.minimum_frames_for_analysis}).")
            return {'is_drowsy': None, 'confidence': 0.0, 'details': {'reason': 'insufficient_frames', 'total_frames': total_frames}}

        # Handle cases where no relevant detections occurred, likely resulting in False
        # Use a small tolerance for normal state % in case of rounding issues
        if total_eye_closed_frames == 0 and yawn_count == 0 and detection_results.get('normal_state_%', 0.0) > 99.0:
             logging.info("No significant drowsiness indicators detected (primarily normal state).")
             details = self._create_details_dict(0, 0, 0, detection_results.get('normal_state_%', 100.0),
                                                 0, 0, 0, 0, 0, 0,
                                                 "no_significant_indicators", 0, 0, 0, 0,
                                                 total_frames, total_frames, fps)
             return {'is_drowsy': False, 'confidence': 0.0, 'details': details}
        elif total_eye_closed_frames == 0 and yawn_count == 0:
             # Handle cases with maybe some unknown state but no eye/yawn
             logging.info("No eye closure or yawn detections found.")
             details = self._create_details_dict(0, 0, 0, detection_results.get('normal_state_%', 0.0),
                                                0, 0, 0, 0, 0, 0, "no_detection",
                                                0, 0, 0, 0, normal_state_frames, total_frames, fps)
             return {'is_drowsy': False, 'confidence': 0.0, 'details': details}


        # --- 2. Calculate Primary Metrics ---
        time_in_seconds = total_frames / fps if fps > 0 else 0
        time_in_minutes = time_in_seconds / 60 if time_in_seconds > 0 else 0

        perclos = detection_results.get('perclos_%', 0.0) # already percentage
        max_closure_duration = detection_results.get('max_closure_duration_s', 0.0)
        yawn_rate_per_minute = detection_results.get('yawn_rate_per_min', 0.0)

        # --- NEW: Cap yawn rate to a realistic maximum ---
        max_realistic_yawn_rate = 20.0 # Adjust if needed based on data
        yawn_rate_per_minute = min(yawn_rate_per_minute, max_realistic_yawn_rate)

        normal_state_percentage = detection_results.get('normal_state_%', 0.0)


        logging.info(f"Metrics: PERCLOS={perclos:.2f}%, Max Closure={max_closure_duration:.2f}s, "
                     f"Yawn Rate={yawn_rate_per_minute:.2f}/min, Normal State={normal_state_percentage:.2f}%")

        # --- 3. Check for Conditional Extreme Overrides ---
        final_score = 0.0
        is_drowsy = False
        reason = "checking_extremes"
        override_triggered = False

        # Check if normal state allows overrides
        allow_override = normal_state_percentage < self.override_max_normal_perc

        if allow_override:
            # Prioritize more reliable extreme indicators
            if max_closure_duration >= self.extreme_duration_threshold:
                reason = f"extreme_duration (>{self.extreme_duration_threshold}s) & low_normal ({normal_state_percentage:.1f}%)"
                override_triggered = True
            elif perclos >= self.extreme_perclos_threshold:
                reason = f"extreme_perclos (>{self.extreme_perclos_threshold}%) & low_normal ({normal_state_percentage:.1f}%)"
                override_triggered = True
            elif yawn_rate_per_minute >= self.extreme_yawn_rate_threshold:
                 reason = f"extreme_yawn_rate (>{self.extreme_yawn_rate_threshold}/min) & low_normal ({normal_state_percentage:.1f}%)"
                 override_triggered = True

            if override_triggered:
                is_drowsy = True
                # Assign a high score, bypassing normal calculation and damping
                final_score = self.score_cap # Use max possible score
                logging.info(f"Conditional Drowsiness Override Triggered: {reason}")
                details = self._create_details_dict(perclos, max_closure_duration, yawn_rate_per_minute, normal_state_percentage,
                                                    0, 0, 0, 0, 0, 0, # Scores/damping not applicable here
                                                    reason, yawn_count, eye_closed_detection_count, total_eye_closed_frames,
                                                    max_consecutive_eye_closed, normal_state_frames, total_frames, fps)
                return {'is_drowsy': True, 'confidence': final_score, 'details': details}
        else:
             # Log only if there were potential indicators that could have triggered override
             potential_override = (max_closure_duration >= self.extreme_duration_threshold or
                                   perclos >= self.extreme_perclos_threshold or
                                   yawn_rate_per_minute >= self.extreme_yawn_rate_threshold)
             if potential_override:
                 logging.info(f"Potential override skipped due to high normal state ({normal_state_percentage:.1f}% >= {self.override_max_normal_perc}%)")


        # --- 4. Calculate Individual Scores (If no override) ---
        perclos_score = self._calculate_metric_score(perclos, self.perclos_threshold, self.perclos_scale)
        duration_score = self._calculate_metric_score(max_closure_duration, self.max_closure_duration_threshold, self.duration_scale)
        yawn_score = self._calculate_metric_score(yawn_rate_per_minute, self.yawn_rate_threshold, self.yawn_rate_scale)

        # Combine eye scores using AVERAGE
        combined_eye_score = (perclos_score + duration_score) / 2.0

        # --- 5. Calculate Raw Drowsiness Score ---
        raw_drowsiness_score = (combined_eye_score * self.eye_metric_weight +
                                yawn_score * self.yawn_metric_weight)

        # --- 6. Apply Non-Linear Normal State Damping ---
        damping_amount = self._calculate_damping(normal_state_percentage)

        final_score = raw_drowsiness_score * (1.0 - damping_amount)
        final_score = max(0.0, min(final_score, self.score_cap)) # Ensure score stays within [0, score_cap]

        # --- 7. Make Final Drowsiness Decision ---
        is_drowsy = final_score >= self.drowsiness_decision_threshold

        # --- 8. Determine Reason ---
        if is_drowsy:
             # Check contributions before damping to see what pushed it over
             eye_contribution = combined_eye_score * self.eye_metric_weight
             yawn_contribution = yawn_score * self.yawn_metric_weight
             # Use a slightly larger tolerance or relative comparison if needed
             if eye_contribution > yawn_contribution + 0.05:
                 reason = f"eye_metrics_dominant (Score: {final_score:.2f})"
             elif yawn_contribution > eye_contribution + 0.05:
                 reason = f"yawn_metrics_dominant (Score: {final_score:.2f})"
             else:
                 # Check if only one metric was non-zero
                 if combined_eye_score > 0 and yawn_score == 0:
                      reason = f"eye_metrics_only (Score: {final_score:.2f})"
                 elif yawn_score > 0 and combined_eye_score == 0:
                      reason = f"yawn_metrics_only (Score: {final_score:.2f})"
                 else:
                      reason = f"combined_metrics_threshold_met (Score: {final_score:.2f})"
        elif final_score > 0:
            reason = f"indicators_present_below_threshold (Score: {final_score:.2f})"
        else:
             # Reached here means final_score is 0
             if raw_drowsiness_score > 0: # Was non-zero before damping
                 reason = f"indicators_damped_to_zero (Raw: {raw_drowsiness_score:.2f}, Damp: {damping_amount:.2f})"
             else: # Raw score was already zero
                 reason = "no_significant_indicators"


        # --- 9. Format Results ---
        details = self._create_details_dict(perclos, max_closure_duration, yawn_rate_per_minute, normal_state_percentage,
                                            perclos_score, duration_score, yawn_score, combined_eye_score,
                                            raw_drowsiness_score, damping_amount, reason,
                                            yawn_count, eye_closed_detection_count, total_eye_closed_frames,
                                            max_consecutive_eye_closed, normal_state_frames, total_frames, fps)

        result = {
            'is_drowsy': is_drowsy,
            'confidence': final_score,
            'details': details
        }

        # Avoid logging every single analysis unless debugging
        # logging.info(f"Analysis result: is_drowsy={result['is_drowsy']}, score={result['confidence']:.3f}, reason={result['details']['reason']}")
        return result

    def _create_details_dict(self, perclos, duration, yawn_rate, normal_perc,
                             p_score, dur_score, y_score, eye_score, raw_score, damping,
                             reason, yawn_cnt, eye_closed_det_cnt, tot_eye_frames, max_consec,
                             norm_frames, tot_frames, fps):
        """Helper function to create the details dictionary."""
        return {
                'perclos_%': perclos,
                'max_closure_duration_s': duration,
                'yawn_rate_per_min': yawn_rate,
                'normal_state_%': normal_perc,
                'perclos_score': p_score,
                'duration_score': dur_score,
                'yawn_score': y_score,
                'combined_eye_score_avg': eye_score,
                'raw_drowsiness_score': raw_score,
                'applied_damping_factor': damping,
                'reason': reason,
                # Raw Inputs
                'yawn_count': yawn_cnt,
                'eye_closed_detection_count': eye_closed_det_cnt,
                'total_eye_closed_frames': tot_eye_frames,
                'max_consecutive_eye_closed_frames': max_consec,
                'normal_state_frames': norm_frames,
                'total_frames': tot_frames,
                'fps': fps
            }


def evaluate_analyzer_from_csv(csv_filepath, analyzer):
    """Evaluates the analyzer against data in a CSV file, using 'Take Type' as ground truth."""
    results_output = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    skipped_rows = 0
    valid_predictions = 0
    processed_rows = 0

    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            # --- CORRECTED: Check for required columns ---
            required_columns = ['Take Type', 'Details']
            if not all(col in csv_reader.fieldnames for col in required_columns):
                 missing = [col for col in required_columns if col not in csv_reader.fieldnames]
                 raise ValueError(f"CSV must contain the following columns: {required_columns}. Missing: {missing}")

            for i, row in enumerate(csv_reader):
                processed_rows = i + 1
                # --- Ground Truth Determination (CORRECTED) ---
                take_type = row.get('Take Type', '').strip()
                if take_type == 'True Alarm':
                    actual_drowsy = True
                # --- Assume anything else is NOT drowsy for this evaluation ---
                # Consider if other 'Take Type' values need specific handling
                # For now, only 'True Alarm' maps to True, everything else to False.
                elif take_type: # If take_type is not empty and not 'True Alarm'
                    actual_drowsy = False
                else:
                    logging.warning(f"Skipping row {i+1}: Empty or missing 'Take Type' field.")
                    skipped_rows += 1
                    continue

                details_json_str = row.get('Details')
                if not details_json_str:
                    logging.warning(f"Skipping row {i+1}: Empty 'Details' field.")
                    skipped_rows += 1
                    continue

                try:
                    detection_results = json.loads(details_json_str)
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping row {i+1}: Invalid JSON in 'Details' field. Error: {e}")
                    skipped_rows += 1
                    continue

                # --- Run Analyzer ---
                try:
                    analyzer_output = analyzer.analyze(detection_results)
                    predicted_drowsy = analyzer_output['is_drowsy']
                except Exception as e:
                    logging.error(f"Error analyzing row {i+1}: {e}")
                    logging.error(f"Problematic Details JSON: {details_json_str[:500]}...") # Log part of the details
                    skipped_rows += 1
                    continue


                # Handle cases where analyzer returns None (e.g., insufficient frames)
                if predicted_drowsy is None:
                     logging.info(f"Row {i+1}: Analyzer returned None ({analyzer_output.get('details', {}).get('reason', 'unknown')}). Skipping metric calculation.")
                     skipped_rows += 1
                     continue

                valid_predictions += 1 # Count only predictions where comparison is possible

                # --- Metric Calculation ---
                is_correct = (predicted_drowsy == actual_drowsy)

                if predicted_drowsy and actual_drowsy:
                    true_positives += 1
                elif predicted_drowsy and not actual_drowsy:
                    false_positives += 1
                    results_output.append({
                        "row_index": i + 1,
                        "row_data": row,
                        "analyzer_output": analyzer_output,
                        "type": "False Positive"
                    })
                elif not predicted_drowsy and not actual_drowsy:
                    true_negatives += 1
                elif not predicted_drowsy and actual_drowsy:
                    false_negatives += 1
                    results_output.append({
                        "row_index": i + 1,
                        "row_data": row,
                        "analyzer_output": analyzer_output,
                        "type": "False Negative"
                    })

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_filepath}'")
        return 0.0, 0.0, 0.0, 0.0, []
    except ValueError as ve:
        print(f"Error: {ve}")
        return 0.0, 0.0, 0.0, 0.0, []
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")
        return 0.0, 0.0, 0.0, 0.0, []


    # --- Calculate Metrics (avoid division by zero) ---
    total_comparisons = true_positives + false_positives + true_negatives + false_negatives
    if total_comparisons == 0:
         print("\n--- Evaluation Summary ---")
         print("No valid comparisons could be made.")
         print(f"Total Rows Processed: {processed_rows}")
         print(f"Skipped Rows: {skipped_rows}")
         return 0.0, 0.0, 0.0, 0.0, []

    accuracy = (true_positives + true_negatives) / total_comparisons if total_comparisons > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0 # TN Rate

    print(f"\n--- Evaluation Summary ---")
    print(f"Total Rows Processed: {processed_rows}")
    print(f"Skipped Rows (Invalid/Missing Data/Analysis Error): {skipped_rows}")
    print(f"Valid Predictions Compared: {valid_predictions}")
    print(f"--------------------------")
    print(f"True Positives (TP): {true_positives}")
    print(f"False Positives (FP): {false_positives}")
    print(f"True Negatives (TN): {true_negatives}")
    print(f"False Negatives (FN): {false_negatives}")
    print(f"--------------------------")
    print(f"Accuracy:    {accuracy:.4f} ((TP + TN) / Total)")
    print(f"Precision:   {precision:.4f} (TP / (TP + FP))")
    print(f"Recall:      {recall:.4f} (Sensitivity, TP / (TP + FN))")
    print(f"Specificity: {specificity:.4f} (TN / (TN + FP))")
    print(f"F1-Score:    {f1_score:.4f}")

    # Sort incorrect results for easier analysis
    results_output.sort(key=lambda x: (x['type'], -x['analyzer_output'].get('confidence', 0)))

    return accuracy, precision, recall, f1_score, results_output


# --- Main Execution ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    # --- Define Parameter Sets ---

    # Original parameters from your code
    params = {
        # --- Basic Thresholds ---
        "perclos_threshold": 15.0,          # Increased slightly
        "max_closure_duration_threshold": 0.4, # Increased slightly
        "yawn_rate_threshold": 3.0,

        # --- Extreme Thresholds for Conditional Overrides ---
        "extreme_perclos_threshold": 45.0,     # Raised
        "extreme_duration_threshold": 1.5,     # Raised
        "extreme_yawn_rate_threshold": 15.0,
        # Normal state MUST be below this for override to trigger
        "override_max_normal_perc": 40.0,   

        # --- Score Calculation Parameters ---
        "perclos_scale": 1.2,           # Reduced scale
        "duration_scale": 1.8,          # Reduced scale
        "yawn_rate_scale": 1.5,
        "score_cap": 2.5,               # Moderate cap

        # --- Weights (Balanced) ---
        "eye_metric_weight": 0.5,
        "yawn_metric_weight": 0.5,

        # --- Non-Linear Damping Parameters ---
        # "Damping" = base * (normal_perc / 100) ^ power
        "damping_base_factor": 0.8, # Max damping effect at 100% normal
        "damping_power": 2.5,       # Power > 1 means steeper increase at high normal %

        # --- Decision Making ---
        "drowsiness_decision_threshold": 0.55, # Adjusted threshold

        # --- Minimum requirements ---
        "minimum_frames_for_analysis": 30, 
        "fps": 20           
    }


    current_params_name = "Iteration (Corrected GT)"
    current_params = params

    analyzer_to_test = RateBasedAnalyzer(**current_params)
    csv_file = 'all_drowsiness_predictions.csv' # Path to your CSV file

    print(f"--- Evaluating Analyzer with Parameters: {current_params_name} ---")
    pp = pprint.PrettyPrinter(indent=2)
    print("Parameters:")
    pp.pprint(current_params)
    print("------------------------------------")


    accuracy, precision, recall, f1_score, incorrect_results = evaluate_analyzer_from_csv(csv_file, analyzer_to_test)

    print("\n--- Detailed Incorrect Predictions ---")
    if not incorrect_results:
        print("No incorrect predictions found!")
    else:
        print(f"Showing details for {len(incorrect_results)} incorrect predictions (FP/FN):")

    # for result in incorrect_results:
    #     print("\n--- Incorrect Case ---")
    #     actual_drowsy_val = result['row_data']['Take Type'] == 'True Alarm' # Re-calculate for print clarity
    #     print(f"Row Index: {result['row_index']}")
    #     print(f"Prediction Type: {result['type']}") # FP or FN
    #     # --- CORRECTED: Show actual ground truth used ---
    #     print(f"Actual Drowsy (from Take Type='True Alarm'?): {actual_drowsy_val} (Take Type: '{result['row_data']['Take Type']}')")
    #     # You might still want to see the original 'Drowsy' column value for context
    #     print(f"Original 'Drowsy' Column Value: {result['row_data'].get('Drowsy', 'N/A')}")
    #     print(f"Event Type: {result['row_data'].get('Event Type', 'N/A')}")
    #     print(f"Memo: {result['row_data'].get('Memo', 'N/A')}")
    #     analyzer_details = result['analyzer_output']['details']
    #     print(f"Analyzer Output: Is Drowsy: {result['analyzer_output']['is_drowsy']}, Confidence: {result['analyzer_output']['confidence']:.3f}")
    #     print(f"  Reason: {analyzer_details['reason']}")
    #     print(f"  Metrics: PERCLOS={analyzer_details['perclos_%']:.1f}%, Duration={analyzer_details['max_closure_duration_s']:.2f}s, YawnRate={analyzer_details['yawn_rate_per_min']:.1f}/min, Normal={analyzer_details['normal_state_%']:.1f}%")
    #     print(f"  Scores: Raw={analyzer_details['raw_drowsiness_score']:.3f}, Eye={analyzer_details['combined_eye_score_avg']:.3f}, Yawn={analyzer_details['yawn_score']:.3f}, Damp={analyzer_details['applied_damping_factor']:.3f}")
        # print(f"Input Details: {result['row_data']['Details']}") # Uncomment for full input JSON
