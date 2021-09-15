import pandas as pd
import numpy as np

from timeseries.processing import remove_following, remove_leading, element_checker


def split_by_holes(series, min_hole_size, min_segment_size, is_nan_hole=True, is_inf_hole=False, hole_values=[]):
    """
    Splits the input pd.DataFrame or pd.Series into segments larger than min_segment_size separated by at least
    min_hole_size hole_elements
    :param series:
    :param min_hole_size:
    :param min_segment_size: minimal number of non hole elements in the segment
    :param hole_elements: elements that will be considered as a "hole" in the dataset
    :return: returns a list of index pairs (index_segment_start, index_segment_end) for each segment. Element at
    index_segment_end is included in the segment. Original index values of the series are returned so please use .loc
    with the results
    """
    assert min_hole_size > 0 and min_segment_size > 0, "Minimum hole and segment size must be greater than 0"

    series = remove_following(series, is_nan_hole, is_inf_hole, hole_values)
    print("After removing following")
    print(series.to_numpy())

    series = remove_leading(series, is_nan_hole, is_inf_hole, hole_values)
    print("After removing leading")
    print(series.to_numpy())

    bool_mask = series.map(lambda x: element_checker(x, is_nan_hole, is_inf_hole, hole_values))

    index_pairs = []
    hole_counter = 0
    segment_counter = 0
    in_segment = False

    hole_start_index = -1
    start_index = 0
    end_index = -1

    def encountered_hole_element():
        nonlocal hole_counter, index_pairs, segment_counter, in_segment, start_index, end_index, i, hole_start_index

        if hole_counter == 0:
            hole_start_index = i

        hole_counter += 1
        if hole_counter >= min_hole_size:
            entered_hole()

    def encountered_non_hole_element():
        nonlocal hole_counter, index_pairs, segment_counter, in_segment, start_index, end_index, i, hole_start_index
        hole_counter = 0
        segment_counter += 1
        if start_index == -1:
            start_index = i

        if segment_counter >= min_segment_size and in_segment is False:
            print(f"Segment {series.iloc[i]}")
            min_segment_recognized()

    def entered_hole():
        nonlocal hole_counter, index_pairs, segment_counter, in_segment, start_index, end_index, i, hole_start_index
        print(f"Hole {series.iloc[i - 1]}")
        if in_segment:
            # end_index = i - 1
            end_index = hole_start_index - 1

            # store actual index values
            index_pairs.append((series.index[start_index], series.index[end_index]))

            in_segment = False
            segment_counter = 0

        start_index = -1

    def min_segment_recognized():
        nonlocal hole_counter, index_pairs, segment_counter, in_segment, start_index, end_index, i, hole_start_index

        # only once per segment
        in_segment = True

    for i, is_nan in enumerate(bool_mask):
        if is_nan:
            encountered_hole_element()
        else:
            encountered_non_hole_element()

    # close up the segment since we've reached the end
    hole_start_index = i + 1
    entered_hole()

    return index_pairs


if __name__ == "__main__":
    series = pd.Series([np.NaN, np.NaN, np.NaN, 1,  2, 3, 4, 5, np.NaN, 5, 7, 9, np.NaN, np.NaN, np.NaN, 10, 11,
                        np.NaN, np.NaN, 12, np.NaN, 13, 14, np.NaN])
    print("Original")
    print(series.to_numpy())
    index_pairs = split_by_holes(series, min_hole_size=2, min_segment_size=2)

    print("After splitting")
    for index_start, index_end in index_pairs:
        print(series.loc[index_start:index_end].to_numpy())
