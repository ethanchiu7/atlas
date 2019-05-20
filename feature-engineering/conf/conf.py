# feature name which is needed in sample flow
# format output by index num

# ---- feature which is needed in sample flow -----

feature_index_dict = {'utdid': 0, 'tm': 1, 'read': 2, 'app': 3, 'ct_lang': 4,
                      'lbs_prov': 5, 'lbs_city': 6, 'act_level': 7, 'user_lt_top_cate': 8, 'stL1C': 9,
                      }


index_feature_dict = {v: k for k, v in feature_index_dict.items()}

# default value
DEFAULT_VALUE = '-1'

# time window (sec)
TIME_WIN = 10 * 60

# positive ratio
POS_RAT = 0.4


# --------------  ASSERT ---------------------

assert max(feature_index_dict.values()) + 1 == len(feature_index_dict)
assert len(feature_index_dict) == len(index_feature_dict) and len(index_feature_dict) > 0

