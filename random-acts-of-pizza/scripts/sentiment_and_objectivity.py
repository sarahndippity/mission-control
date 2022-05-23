import pandas as pd
from textblob import TextBlob


def sentiment(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one new column in training dataset: sentiment of the post.
    
    Function takes in a pandas Dataframe as input, and returns the same Dataframe
    with the 1 additional column.
    """
    df = train_df.copy()

    # use textblob to assess sentiment in text
    text_blob = [TextBlob(text) for text in df['title_and_request']]
    df['text_polarity'] = [x.sentiment.polarity for x in text_blob]
    df['sentiment'] = (df['text_polarity'].apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0)))
    
    # drop unnecessary columns
    df.drop('text_polarity', inplace=True, axis=1)
    
    return df


def subjectivity(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one new column in training dataset: objectivity of the post.
    
    Function takes in a pandas Dataframe as input, and returns the same Dataframe
    with the 1 additional column.
    """
    df = train_df.copy()

    # use textblob to assess subjectivity in text
    text_blob = [TextBlob(text) for text in df['title_and_request']]
    df['text_subjectivity'] = [x.sentiment.subjectivity for x in text_blob]
    df['subjectivity'] = (df['text_subjectivity'].apply(lambda x: 1 if x > 0.5 else 0))
    
    # drop unnecessary column
    df.drop('text_subjectivity', inplace=True, axis=1)
    
    return df