import pandas as pd


#create narr dict
def create_dict():
    narr_dict = {'money': """money now broke week until time last day 
    when today tonight paid next
    first night after tomorrow month while
    account before long Friday rent buy
    bank still bills bills ago cash due due
    soon past never paycheck check spent
    years poor till yesterday morning dollars
    financial hour bill evening credit
    budget loan bucks deposit dollar current
    payed""",
                'job' : """work job paycheck unemployment interview
    fired employment hired hire""",
                'student' : """college student school roommate
    studying university finals semester
    class study project dorm tuition""",
                'family':"""family mom wife parents mother husband
    dad son daughter father parent
    mum""", 
                'craving':"""friend girlfriend craving birthday
    boyfriend celebrate party game games
    movie date drunk beer celebrating invited
    drinks crave wasted invite"""}

    #len_data = len(dataset)

    money_frqs = []
    job_frqs = []
    student_frqs = []
    family_frqs = []
    craving_frqs = []

    for key in narr_dict.keys():
        narr_dict[key] = set(narr_dict[key].lower().replace('\n',' ').split())
    
    return narr_dict

def word_count(post, narr, narr_dict):
    
    #initialize variables
    len_text = len(post) + 1
    count = 0
    
    for word in post:
        count += 1 if word in narr_dict[narr] else 0
        
    #normalize counts by text length
    count /= len_text
    
    return count
        
def narr_assign(df: pd.DataFrame) -> pd.DataFrame:
    
    #should probably caluclat narr dict here
    narr_dict = create_dict()
    #copy 
    df2 = df.copy()
    
    #extract all texts
    texts = [x.lower().replace('\n','').replace('.',' ').split() for 
             x in df['request_text_edit_aware']]
    #count words in each post for the narr
    for key in narr_dict.keys():
        df2[key] = [word_count(x,key,narr_dict) for x in texts] #row wise function to count elements
    
    #caluclate percentiles and assign to dict
    percs = {}
    for key in narr_dict.keys():
        df2[key+'_quant'] = df2[key].quantile(.75)
        
    #now compare money val with quants to assign 0,1
    for key in narr_dict.keys():
        df2[key+'_check'] = 1 * (df2[key] >= df2[key+'_quant'])
    
    #drop extra columns
    # drop unnecessary columns
    for key in narr_dict.keys():
        df2.drop(key, inplace=True, axis=1)
        df2.drop(key+'_quant', inplace=True, axis=1)
        
    
    return df2

