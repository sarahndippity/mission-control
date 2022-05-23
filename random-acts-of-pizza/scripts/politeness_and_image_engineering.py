#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 12:02:56 2021

@author: boneil
"""


import numpy as np
import pandas as pd


image_text = "imgur.com|.jpg|.png|.jpeg"
reciprocity_text = "pay it forward|pay it back|return the favor"
polite_text = "please|thank|thanks"


def has_text_binary_column_creator(train_df: pd.DataFrame, text_to_search: str, column_name: str) -> pd.DataFrame:
    """
    Create one new column in training dataset: 1 if string contained in post text, 0 if not
    
    Function takes in a pandas Dataframe as input, text to search and column name.
    Returns the same Dataframe with the 1 additional columns.
    
    """
    df = train_df.copy()
    
    # length of original post
    df[column_name] = (df["title_and_request"].str.contains(text_to_search, case=False).apply(lambda x : int(x)))
    
    return df

### Implementation Code
#import politeness_and_image_engineering as text_bi

#text_bi.has_text_binary_column_creator(train_df,image_text,"has_image")
#text_bi.has_text_binary_column_creator(train_df,reciprocity_text,"has_reciprocity")
#text_bi.has_text_binary_column_creator(train_df,polite_text,"has_polite")


