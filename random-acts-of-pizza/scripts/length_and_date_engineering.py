from datetime import datetime as dt
import numpy as np
import pandas as pd
import re

def create_length_of_post_and_title(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create two new columns in training dataset: length of original post and
    length of title of post.
    
    Function takes in a pandas Dataframe as input, and returns the same Dataframe
    with the 2 additional columns.
    """
    df = train_df.copy()
    
    # length of original post
    df["length_of_original_post"] = df["request_text_edit_aware"].apply(len)

    # length of title
    df["length_of_title"] = df["request_title"].apply(len)
    
    return df

def create_title_pentagram(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns in training dataset from first five words of title.
    
    Function takes in a pandas Dataframe as input, and returns the same Dataframe
    with the 1 additional column.
    """
    df = train_df.copy()
    
    # first remove punctuation and extra spaces from title before extracting first 5 words
    df["title_pentagram"] = (df["request_title"]
                             .apply(lambda s: re.sub("request", "", s.lower()))
                             .apply(lambda s: re.sub("[^\w\s]", "", s))
                             .apply(lambda s: re.sub("\s+", " ", s))
                             .apply(lambda s: re.sub("^\s", "", s))
                             .apply(lambda x: x.split(" ")[:5])
                            )
    
    return df
    
def create_date_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns using the date/timestamp feature in the original dataset.
    day_of_week: Number representing the day of the week. 0 is Monday, 1 is Tuesday...
    month_of_year: Number representing the month in the year.
    time_bucket: String column telling what time of day the original request was posted.
        This is recorded in the original poster's timezone, NOT UTC.
    
    Function takes in a pandas Dataframe as input, and returns the same Dataframe
    with the 3 additional columns.
    """
    df = train_df.copy()
    
    df["request_timestamp_datetime"] = df["unix_timestamp_of_request"].apply(dt.fromtimestamp)
    df["day_of_week"] = df["request_timestamp_datetime"].apply(dt.weekday)
    df["month_of_year"] = df["request_timestamp_datetime"].apply(dt.date).apply(lambda x: x.month)
    df["time_bucket"] = (df["request_timestamp_datetime"].apply(dt.time).apply(lambda x: x.hour)
                         .apply(time_of_day_selection))
    
    df.drop("request_timestamp_datetime", axis=1, inplace=True)
    
    return df
    
def time_of_day_selection(t: pd.Series) -> pd.Series:
    """
    Helper function to calculate time of day from datetime object.
    
    Function takes in a pandas Series object (or pd.Dataframe column) as input and
    returns a Series object.
    """
    # set of conditions
    condlist = [np.logical_and(t >= 0, t < 6), 
                np.logical_and(t >= 6, t < 12), 
                np.logical_and(t >= 12, t < 18), 
                np.logical_and(t >= 18, t < 24)]
    # set of results
    choicelist = ["overnight", "morning", "afternoon", "evening"]
    # apply the conditions to the input Series
    converted_list = np.select(condlist, choicelist)
    
    return converted_list
    
def create_deltas(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    
    # time elapsed from request to retrieval
    df["time_elapsed"] = df["requester_account_age_in_days_at_retrieval"] - df["requester_account_age_in_days_at_request"]
    # requester comments
    df["change_in_number_requester_comments"] = df["requester_number_of_comments_at_retrieval"] - df["requester_number_of_comments_at_request"]
    # requester comments in raop
    df["change_in_number_requester_comments_raop"] = df["requester_number_of_comments_in_raop_at_retrieval"] - df["requester_number_of_comments_in_raop_at_request"]
    # requester posts
    df["change_in_number_requester_posts"] = df["requester_number_of_posts_at_retrieval"] - df["requester_number_of_posts_at_request"]
    # requester posts in raop
    df["change_in_number_requester_posts_raop"] = df["requester_number_of_posts_on_raop_at_retrieval"] - df["requester_number_of_posts_on_raop_at_request"]
    # voter status (requester's upvotes minus downvotes, indication of how positive of a person they likely are)
    df["change_in_requester_vote_status"] = df["requester_upvotes_minus_downvotes_at_retrieval"] - df["requester_upvotes_minus_downvotes_at_request"]
    # engagement status (requester's upvotes plus downvotes, indication of how actively they engage with other Reddit posts)
    df["change_in_requester_engagement"] = df["requester_upvotes_plus_downvotes_at_retrieval"] - df["requester_upvotes_plus_downvotes_at_request"]
    # percentage of upvotes & downvotes of request at retrieval (indication of post popularity)
    df["total_request_votes_at_retrieval"] = df["number_of_downvotes_of_request_at_retrieval"] + df["number_of_upvotes_of_request_at_retrieval"]
    df["percent_request_upvotes_at_retrieval"] = df["number_of_upvotes_of_request_at_retrieval"] / df["total_request_votes_at_retrieval"]
    # replace infinite values with 0
    df = df.replace(np.inf, 0)
    
    return df
