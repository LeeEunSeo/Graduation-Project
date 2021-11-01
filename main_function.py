from sklearn.feature_extraction.text import TfidfVectorizer
import firebase_admin
from firebase_admin import credentials

from firebase_admin import db

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


cred = credentials.Certificate('gfi-rec-firebase-adminsdk-j8kaf-b945a590a2.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://gfi-rec-default-rtdb.firebaseio.com/'
})


def ssd_rtdb(user_id):

    prod_id = db.reference().child('rec_choice').child(user_id).child('prod_id').get()
    min_price = db.reference().child('rec_choice').child(user_id).child('min_price').get()
    max_price = db.reference().child('rec_choice').child(user_id).child('max_price').get()
    purpose_category = db.reference().child('rec_choice').child(user_id).child('purpose_category').get()
    # 1 - ë¬¸ì„œìž‘ì—…  2 - ê°•ì˜  3 - 5 ê²Œìž„(ì €,ì¤‘,ê³ )  6 - ì˜ìƒíŽ¸ì§‘  7 - 3Dê·¸ëž˜í”½  8 - í”„ë¡œê·¸ëž˜ë°

    # íŒŒì´ì–´ë² ì´ìŠ¤ ë°ì´í„° ë¶ˆëŸ¬ì˜¤ê¸°
    ref = db.reference().child(prod_id)
    # Create a query against the collection
    # ref.where(u'price', u'>=', v2).where(u'price', u'<=', v3).stream()
    rows = ref.get()  # ë°ì´í„° ì €ìž¥
    # colname = ref.description
    col = {'img_url', 'price', 'prod_code', 'prod_id', 'prod_info', 'prod_name'}
    # ì»¬ëŸ¼ëª… ì¶”ì¶œí•´ì„œ ì»¬ëŸ¼ ë¦¬ìŠ¤íŠ¸ ìƒì„±
    # for i in colname: col.append(i[0])

    # ë°ì´í„° í”„ë ˆìž„ ìƒì„±í•´ì„œ ê°’ ë„£ê¸°
    raw = pd.DataFrame(list(rows), columns=col)
    # raw[['prod_code', 'prod_id', 'prod_name', 'img_url', 'prod_info', 'price']]
    raw['price'] = pd.to_numeric(raw['price'])

    # raw.drop(raw.loc[raw['price']<v2].index, inplace=True)
    # raw.drop(raw.loc[raw['price']>v3].index, inplace=True)

    raw_len = len(raw)

    result = []
    for item in raw['prod_info']:
        result.append(item)

    num = len(result)

    # ê°€ìƒìœ ì € ë”•ì…”ë„ˆë¦¬ - SSD
    virture_user = {
        1: 'ë‚´ìž¥í˜•SSD 64cm25í˜• SATA3 6Gb/s ìˆœì°¨ì½ê¸° 555MB/s ìˆœì°¨ì“°ê¸° 540MB/s ì½ê¸°IOPS 79K ì“°ê¸°IOPS 87K MTBF 150ë§Œì‹œê°„ A/Sê¸°ê°„ 3ë…„ ë‘ê»˜ 7mm 46g',
        2: 'ë‚´ìž¥í˜•SSD 64cm25í˜• SATA3 6Gb/s ìˆœì°¨ì½ê¸° 560MB/s ìˆœì°¨ì“°ê¸° 510MB/s ì½ê¸°IOPS ìµœëŒ€ 95K  ì“°ê¸°IOPS ìµœëŒ€ 90K MTBF 180ë§Œì‹œê°„ A/Sê¸°ê°„ 5ë…„  ë‘ê»˜ 7mm',
        3: 'ë‚´ìž¥í˜•SSD 64cm25í˜• SATA3 6Gb/s ìˆœì°¨ì½ê¸° 560MB/s ìˆœì°¨ì“°ê¸° 510MB/s ì½ê¸°IOPS ìµœëŒ€ 95K  ì“°ê¸°IOPS ìµœëŒ€ 90K MTBF 180ë§Œì‹œê°„ A/Sê¸°ê°„ 5ë…„  ë‘ê»˜ 7mm',
        4: 'ë‚´ìž¥í˜•SSD M2 2280 PCIe40x4 64GT/s NVMe ìˆœì°¨ì½ê¸° 4900MB/s ìˆœì°¨ì“°ê¸° 400MB/s ì½ê¸°IOPS 750K ì“°ê¸°IOPS 700K MTBF 170ë§Œì‹œê°„ A/Sê¸°ê°„ 3ë…„ ë‘ê»˜ 157mm 45g',
        5: 'ë‚´ìž¥í˜•SSD 64cm25í˜• SATA3 6Gb/s ìˆœì°¨ì½ê¸° 560MB/s ìˆœì°¨ì“°ê¸° 510MB/s ì½ê¸°IOPS ìµœëŒ€ 95K  ì“°ê¸°IOPS ìµœëŒ€ 90K MTBF 180ë§Œì‹œê°„ A/Sê¸°ê°„ 5ë…„  ë‘ê»˜ 7mm',
        6: 'ë‚´ìž¥í˜•SSD M2 2280 PCIe30x4 32GT/s NVMe 13 ìˆœì°¨ì½ê¸° 2300MB/s ìˆœì°¨ì“°ê¸° 900MB/s  MTBF 180ë§Œì‹œê°„ A/Sê¸°ê°„ 3ë…„ ë°ì´í„° ë³µêµ¬ 1ë…„ ë‘ê»˜ 215mm 7g',
        7: 'ë‚´ìž¥í˜•SSD 64cm25í˜• SATA36Gb/s ìˆœì°¨ì½ê¸° 535MB/s  ìˆœì°¨ì“°ê¸° ìµœëŒ€ 370MB/s ì½ê¸°IOPS ìµœëŒ€ 87K  ì“°ê¸°IOPS ìµœëŒ€ 70K A/Sê¸°ê°„ 3ë…„ ë‘ê»˜ 7mm',
        8: 'ë‚´ìž¥í˜•SSD 64cm25í˜• SATA3 6Gb/s ìˆœì°¨ì½ê¸° 560MB/s ìˆœì°¨ì“°ê¸° 510MB/s ì½ê¸°IOPS ìµœëŒ€ 95K  ì“°ê¸°IOPS ìµœëŒ€ 90K MTBF 180ë§Œì‹œê°„ A/Sê¸°ê°„ 5ë…„  ë‘ê»˜ 7mm'
    }

    # ê°€ìƒìœ ì € ì¶”ê°€
    add_vu = virture_user[int(purpose_category)]
    result.append(add_vu)

    vect = CountVectorizer()
    countvect = vect.fit_transform(result)
    countvect.toarray()
    sorted(vect.vocabulary_)
    countvect_df = pd.DataFrame(countvect.toarray(), columns=sorted(vect.vocabulary_))
    cosine_matrix = cosine_similarity(countvect_df, countvect_df)
    np.round(cosine_matrix, 4)  # ì›í•˜ëŠ” ì†Œìˆ˜ì  ìžë¦¬ìˆ˜ì—ì„œ ë°˜ì˜¬ë¦¼
    vect = TfidfVectorizer()
    tfvect = vect.fit(result)
    tfidv_df = pd.DataFrame(tfvect.transform(result).toarray(), columns=sorted(vect.vocabulary_))

    for column_name, item in tfidv_df.iteritems():
        if 'TB' in column_name:
            tfidv_df[column_name] = item + 0.01
        if 'tb' in column_name:
            tfidv_df[column_name] = item + 0.01
        if 'mb' in column_name:
            tfidv_df[column_name] = item + 0.008
        if 'MB' in column_name:
            tfidv_df[column_name] = item + 0.006
        if 'RPM' in column_name:
            tfidv_df[column_name] = item + 0.004
        if 'rpm' in column_name:
            tfidv_df[column_name] = item + 0.004
        if 'ë…„' in column_name:
            tfidv_df[column_name] = item + 0.002

    cosine_matrix = cosine_similarity(tfidv_df, tfidv_df)
    vect = TfidfVectorizer(max_features=5)
    tfvect = vect.fit(result)
    tfidv_df = pd.DataFrame(tfvect.transform(result).toarray(), columns=sorted(vect.vocabulary_))

    pro2id = {}
    for i, c in enumerate(raw['prod_name']): pro2id[i] = c
    pro2id[num] = 'virture_user'
    id2pro = {}
    for i, c in pro2id.items(): id2pro[c] = i

    # id ì¶”ì¶œ
    idx = id2pro['virture_user']  # ê°€ìƒìœ ì € - ë§ˆì§€ë§‰ ì¸ë±ìŠ¤
    sim_scores = [(i, c) for i, c in enumerate(cosine_matrix[idx]) if
                  i != idx]  # ì œí’ˆë“¤ì˜ ìœ ì‚¬ë„ ë° ì¸ë±ìŠ¤ë¥¼ ì¶”ì¶œ
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # ìœ ì‚¬ë„ê°€ ë†’ì€ ìˆœì„œëŒ€ë¡œ ì •ë ¬
    # sim_scores[0:10]  # ìƒìœ„ 10ê°œì˜ ì¸ë±ìŠ¤ì™€ ìœ ì‚¬ë„ë¥¼ ì¶”ì¶œ

    # ì¸ë±ìŠ¤ë¥¼ Titleë¡œ ë³€í™˜
    sim_scores = [(pro2id[i], score) for i, score in sim_scores[0:10]]

    rec_ref = db.reference().child('rec').child(user_id)
    rec_ref.set({'1': sim_scores[0][0],
                 '2': sim_scores[1][0],
                 '3': sim_scores[2][0],
                 '4': sim_scores[3][0],
                 '5': sim_scores[4][0],
                 '6': sim_scores[5][0],
                 '7': sim_scores[6][0],
                 '8': sim_scores[7][0],
                 '9': sim_scores[8][0],
                 '10': sim_scores[9][0]})
