from sklearn.feature_extraction.text import TfidfVectorizer
import firebase_admin
from firebase_admin import credentials

from firebase_admin import db

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cred = credentials.Certificate('goforit-af276-firebase-adminsdk-b3izz-b61829e785.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://goforit-af276-default-rtdb.firebaseio.com/'
})


def ssd_rtdb(user_id):
    prod_id = db.reference().child('rec_choice').child(user_id).child('prod_id').get()
    # min_price = db.reference().child('rec_choice').child(user_id).child('min_price').get()
    max_price = db.reference().child('rec_choice').child(user_id).child('max_price').get()
    purpose_category = db.reference().child('rec_choice').child(user_id).child('purpose_category').get()
    if prod_id:
        # 1 - ë¬¸ì„œìž‘ì—…  2 - ê°•ì˜  3 - 5 ê²Œìž„(ì €,ì¤‘,ê³ )  6 - ì˜ìƒíŽ¸ì§‘  7 - 3Dê·¸ëž˜í”½  8 - í”„ë¡œê·¸ëž˜ë°
        prod_id = str(prod_id).lower() + "list"
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

        raw.drop(raw.loc[raw['price'] < int(max_price)].index,
                 inplace=True)  # max_price보다 큰 값을 price로 가지고 있는 행을 삭제하고 싶습니다.

        raw_len = len(raw)

        result = []
        for item in raw['prod_info']:
            result.append(item)

        num = len(result)

        if prod_id == "notebooklist":
            virture_user = {
                1: '3962cm156인치 코어i511세대 쿼드코어 운영체제OS윈도우10 1920x1080FHD DDR4 8GB 256GB 내장그래픽 80211acWiFi5 숫자키패드 배터리65Wh 충전단자DC 두께156mm 무게110kg 용도사무인강용휴대용',
                2: '3962cm156인치 코어i38세대 쿼드코어 운영체제OS윈도우10 1920x1080FHD DDR4 8GB 256GB 내장그래픽 80211acWiFi5 숫자키패드 배터리68Wh 충전단자DC 두께168mm 무게109kg 용도사무인강용휴대용',
                3: '3962cm156인치 코어i38세대 쿼드코어 운영체제OS윈도우10 1920x1080FHD DDR4 8GB 256GB 내장그래픽 80211acWiFi5 숫자키패드 배터리68Wh 충전단자DC 두께168mm 무게125kg 용도사무인강용휴대용',
                4: '4394cm173인치 코어i79세대 헥사코어 운영체제OS 1920x1080FHD 주사율120Hz DDR4 16GB 512GB 외장그래픽 80211acWiFi5 숫자키패드4열 배터리50Wh 충전단자DC 두께262mm 무게285kg 용도게임용그래픽작업용',
                5: '4300cm16인치AMD 라이젠74세대 코어i9세대 옥타코어 운영체제OS미포함프리도스 2560x1600 sRGB100 주사율165Hz DDR4 32GB 512GB 외장그래픽 VRAM8GB 80211acWiFi6 배터리66Wh 두께258mm 무게24kg 용도게임용그래픽작업용',
                6: '4394cm173인치AMD 라이젠94세대 코어i911세대 옥타코어 운영체제OS미포함프리도스 2560x1440QHD 주사율165Hz DDR4 16GB 512GB 외장그래픽 VRAM12GB 80211axWiFi6 숫자키패드4열 배터리80Wh 충전단자DC 두께275mm 무게28kg 용도게임용그래픽작업용',
                7: '3962cm156인치 코어i79세대 헥사코어 운영체제OS미포함프리도스 1920x1080FHD DDR4 8GB 512GB 외장그래픽 80211acWiFi5 배터리66Wh 두께258mm 무게24kg 용도게임용그래픽작업용',
                8: '4394cm173인치 코어i79세대 라이젠74세대 헥사코어 옥타코어 운영체제OS윈도우 미포함프리도스 1920x1080FHD 주사율120Hz DDR4 16GB 512GB 내장그래픽 80211acWiFi5 숫자키패드4열 배터리60Wh 충전단자DC 두께262mm 무게255kg 용도게임용그래픽작업용'
            }
        if prod_id == "cpulist":
            virture_user = {
                1: '인텔소켓1200 AMD소켓AM4 6코어 12쓰레드 기본클럭26GHz 최대클럭44GHz L3캐시12MB TDP65W 3200MHz 내장그래픽탑재 쿨러인텔기본쿨러포함',
                2: '인텔소켓1200 AMD소켓AM4 6코어 12쓰레드 기본클럭26GHz 최대클럭44GHz L3캐시12MB TDP65W 3200MHz 내장그래픽탑재 쿨러인텔기본쿨러포함',
                3: '인텔소켓1200 AMD소켓AM4 6코어 12쓰레드 기본클럭26GHz 최대클럭44GHz L3캐시12MB TDP65W 3200MHz 내장그래픽탑재 쿨러인텔기본쿨러포함',
                4: '인텔소켓1200 AMD소켓AM4 6코어 12쓰레드 기본클럭26GHz 최대클럭44GHz L3캐시12MB TDP65W 3200MHz 내장그래픽탑재 쿨러인텔기본쿨러포함',
                5: '인텔소켓1200 AMD소켓sTRX4 10코어 16쓰레드 기본클럭32GHz 최대클럭50GHz L3캐시16MB TDP165W 3200MHz 내장그래픽탑재',
                6: '인텔소켓2066 AMD소켓sTRX4 14코어 28쓰레드 기본클럭33GHz 최대클럭46GHz L3캐시1925MB TDP165W 3000MHz 내장그래픽미탑재 쿨러미포함',
                7: '인텔소켓2066 AMD소켓sTRX4 14코어 28쓰레드 기본클럭33GHz 최대클럭46GHz L3캐시1925MB TDP165W 3000MHz 내장그래픽미탑재 쿨러미포함',
                8: '인텔소켓1200 AMD소켓sTRX4 10코어 16쓰레드 기본클럭32GHz 최대클럭50GHz L3캐시16MB TDP165W 3200MHz 내장그래픽탑재'
            }
        if prod_id == "mainboard":
            virture_user = {
                1: '인텔소켓1200 인텔 H510 MATX 236x202cm VGA 연결 PCIe40x16 그래픽 출력 DSUB HDMI PCIe 슬롯 2개 M2 1개 SATA3 4개 기가비트 LAN',
                2: '인텔소켓1200 인텔 H510 MATX 236x202cm VGA 연결 PCIe40x16 그래픽 출력 DSUB HDMI PCIe 슬롯 2개 M2 1개 SATA3 4개 기가비트 LAN',
                3: '인텔소켓1155 인텔 Z68 일반ATX305x244cm 전원부 12페이즈 VGA 연결 PCIe30x16 그래픽 출력 DSUB DVI HDMI  SATA2 4개 기가비트 LAN SATA 4개 SATA 6Gbps 2개전원부 84페이즈',
                4: '인텔소켓1200 인텔 H510 MATX 236x202cm VGA 연결 PCIe40x16 그래픽 출력 DSUB HDMI PCIe 슬롯 2개 M2 1개 SATA3 4개 기가비트 LAN',
                5: 'AMD소켓AM4 AMDA520 MATX226x221cm VGA 연결 PCIe30x16 그래픽 출력 DSUB HDMI PCIe 슬롯 3개 M2 1개 SATA3 4개 기가비트 LAN',
                6: 'AMD소켓AM4 AMDA520 MATX226x221cm VGA 연결 PCIe30x16 그래픽 출력 DSUB HDMI PCIe 슬롯 3개 M2 1개 SATA3 4개 기가비트 LAN',
                7: '인텔소켓1200 인텔 H510 MATX 236x202cm VGA 연결 PCIe40x16 그래픽 출력 DSUB HDMI PCIe 슬롯 2개 M2 1개 SATA3 4개 기가비트 LAN',
                8: '인텔소켓1200 인텔 H510 MATX 236x202cm VGA 연결 PCIe40x16 그래픽 출력 DSUB HDMI PCIe 슬롯 2개 M2 1개 SATA3 4개 기가비트 LAN'
            }
        if prod_id == "gpulist":
            virture_user = {
                1: '부스트클럭1800MHz 스트림프로세서3500개 GDDR6DDR6 백플레이트 AS3년',
                2: '부스트클럭1830MHz 스트림프로세서3500개 GDDR6DDR6 백플레이트 AS3년',
                3: '부스트클럭1710MHz 스트림프로세서4864개 GDDR6DDR6 사용전력최대220W 백플레이트 AS3년',
                4: '베이스클럭1550MHz  부스트클럭1770MHz 스트림프로세서6000개 GDDR6XDDR6X 사용전력최대200W 백플레이트 AS3년',
                5: '베이스클럭1400MHz 부스트클럭1770MHz 스트림프로세서10240개 GDDR6XDDR6X GDDR6DDR6 사용전력최대125W 백플레이트 AS3년',
                6: 'RTX3090 부스트클럭1695MHz 스트림프로세서10496개 GDDR6XDDR6X 사용전력최대350W 백플레이트 AS3년',
                7: '부스트클럭1770MHz 스트림프로세서8704개 GDDR6XDDR6X 사용전력최대340W 백플레이트 AS3년',
                8: '베이스클럭1400MHz 부스트클럭1770MHz 스트림프로세서8000개 GDDR6XDDR6X GDDR6DDR6 사용전력최대125W 백플레이트 AS3년'
            }
        if prod_id == "memorylist":
            virture_user = {
                1: '데스크탑용 DDR4 8GB 램개수 ',
                2: '데스크탑용 DDR4 8GB 램개수 2개',
                3: '데스크탑용 DDR3 4GB 램개수 4개',
                4: '데스크탑용 DDR4 8GB 램개수 2개',
                5: '데스크탑용 DDR4 16GB 램개수 2개',
                6: '데스크탑용 DDR4 16GB 램개수 2개',
                7: '데스크탑용 DDR4 8GB 램개수 2개',
                8: '데스크탑용 DDR4 8GB 램개수 2개'
            }
        if prod_id == "ssdlist":
            virture_user = {
                1: '내장형SSD 64cm25형 SATA3 6Gb/s 순차읽기 555MB/s 순차쓰기 540MB/s 읽기IOPS 79K 쓰기IOPS 87K MTBF 150만시간 A/S기간 3년 두께 7mm 46g',
                2: '내장형SSD 64cm25형 SATA3 6Gb/s 순차읽기 560MB/s 순차쓰기 510MB/s 읽기IOPS 최대 95K  쓰기IOPS 최대 90K MTBF 180만시간 A/S기간 5년  두께 7mm',
                3: '내장형SSD 64cm25형 SATA3 6Gb/s 순차읽기 560MB/s 순차쓰기 510MB/s 읽기IOPS 최대 95K  쓰기IOPS 최대 90K MTBF 180만시간 A/S기간 5년  두께 7mm',
                4: '내장형SSD M2 2280 PCIe40x4 64GT/s NVMe 순차읽기 4900MB/s 순차쓰기 400MB/s 읽기IOPS 750K 쓰기IOPS 700K MTBF 170만시간 A/S기간 3년 두께 157mm 45g',
                5: '내장형SSD 64cm25형 SATA3 6Gb/s 순차읽기 560MB/s 순차쓰기 510MB/s 읽기IOPS 최대 95K  쓰기IOPS 최대 90K MTBF 180만시간 A/S기간 5년  두께 7mm',
                6: '내장형SSD M2 2280 PCIe30x4 32GT/s NVMe 13 순차읽기 2300MB/s 순차쓰기 900MB/s  MTBF 180만시간 A/S기간 3년 데이터 복구 1년 두께 215mm 7g',
                7: '내장형SSD 64cm25형 SATA36Gb/s 순차읽기 535MB/s  순차쓰기 최대 370MB/s 읽기IOPS 최대 87K  쓰기IOPS 최대 70K A/S기간 3년 두께 7mm',
                8: '내장형SSD 64cm25형 SATA3 6Gb/s 순차읽기 560MB/s 순차쓰기 510MB/s 읽기IOPS 최대 95K  쓰기IOPS 최대 90K MTBF 180만시간 A/S기간 5년  두께 7mm'
            }
        if prod_id == "hddlist":
            virture_user = {
                1: 'HDD PC용 89cm35인치 1TB SATA3 6Gb/s 7200RPM 메모리 64MB chleo 210MB/s 두께 202mm A/S 정보 2년 무게 최대 400g',
                2: 'HDD PC용 89cm35인치 1TB SATA3 6Gb/s 7200RPM 메모리 64MB chleo 210MB/s 두께 202mm A/S 정보 2년 무게 최대 400g',
                3: '순차읽기 560MB/s 순차쓰기 510MB/s 읽기IOPS 최대 95K 쓰기IOPS 최대 90K MTBF 180만시간 A/S기간 5년 두께 7mm',
                4: 'HDD PC용 89cm35인치 1TB SATA3 6Gb/s 7200RPM 메모리 64MB 기록방식 CMRPMR 두께 261mm A/S 정보 2년 충격 감지센서 램프로딩 기술',
                5: 'HDD PC용 89cm35인치 2TB SATA3 6Gb/s 7200RPM 메모리 64MB chleo 210MB/s 두께 202mm A/S 정보 2년 무게 최대 400g',
                6: ' HDD PC용 1TB SATA 무게 2kg',
                7: 'HDD PC용 89cm35인치 1TB SATA3 6Gb/s 7200RPM 메모리 64MB chleo 210MB/s 두께 202mm A/S 정보 2년 무게 최대 400g',
                8: 'HDD PC용 89cm35인치 1TB SATA3 6Gb/s 7200RPM 메모리 64MB chleo 210MB/s 두께 202mm A/S 정보 2년 무게 최대 400g'
            }
        if prod_id == "powerlist":
            virture_user = {
                1: 'ATX파워 정격출력650W 12V싱글레일 12V가용률100 액티브PFC AS무상6년 메인전원24핀 보조전원84핀 PCIe8핀622개 SATA6개 IDE4핀4개 대기전력1W미만 플랫케이블',
                2: 'ATX파워 정격출력750W 12V가용률99 액티브PFC AS무상7년 메인전원24핀204 보조전원8442 PCIe8핀624개 SATA8개 IDE4핀3개 대기전력1W미만 플랫케이블',
                3: 'ATX파워 정격출력750W 12V가용률99 액티브PFC AS무상7년 메인전원24핀204 보조전원8442 PCIe8핀624개 SATA8개 IDE4핀3개 대기전력1W미만 플랫케이블',
                4: 'ATX파워 정격출력750W 12V싱글레일 12V가용률100 액티브PFC AS무상7년 메인전원24핀204 보조전원844 PCIe8핀623개 SATA8개 IDE4핀5개 부가기능팬리스모드 대기전력1W미만 플랫케이블',
                5: 'ATX파워 정격출력950W 12V싱글레일 12V가용률100 액티브PFC AS무상7년 메인전원24핀 보조전원8핀442개 PCIe8핀624개 SATA8개 IDE4핀4개 FDD 부가기능팬리스모드 대기전력1W미만 플랫케이블',
                6: 'ATX파워 정격출력950W 12V싱글레일 12V가용률100 액티브PFC AS무상7년 메인전원24핀 보조전원8핀442개 PCIe8핀624개 SATA8개 IDE4핀4개 FDD 부가기능팬리스모드 대기전력1W미만 플랫케이블',
                7: 'ATX파워 정격출력1000W 12V싱글레일 12V가용률100 액티브PFC AS무상7년 메인전원24핀 보조전원8핀442개 PCIe8핀628개 SATA10개 IDE4핀5개 FDD 부가기능팬리스모드 대기전력1W미만 플랫케이블',
                8: 'ATX파워 정격출력750W 12V싱글레일 12V가용률100 액티브PFC AS무상7년 메인전원24핀204 보조전원844 PCIe8핀623개 SATA8개 IDE4핀5개 부가기능팬리스모드 대기전력1W미만 플랫케이블'
            }

        # ê°€ìƒìœ ì € ì¶”ê°€
        if purpose_category:
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

        if prod_id == "notebooklist":
            for column_name, item in tfidv_df.iteritems():
                if '코어' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if '윈도우' in column_name:
                    tfidv_df[column_name] = item + 0.008
                if '배터리' in column_name:
                    tfidv_df[column_name] = item + 0.006
                if '휴대용' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if '무게' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if '숫자키패드' in column_name:
                    tfidv_df[column_name] = item + 0.002
        if prod_id == "cpulist":
            for column_name, item in tfidv_df.iteritems():
                if '인텔' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if 'AMD' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if '코어' in column_name:
                    tfidv_df[column_name] = item + 0.008
                if '쓰레드' in column_name:
                    tfidv_df[column_name] = item + 0.006
                if '클럭' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if '내장' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if 'MHz' in column_name:
                    tfidv_df[column_name] = item + 0.002
        if prod_id == "mainboard":
            for column_name, item in tfidv_df.iteritems():
                if '인텔' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if 'AMD' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if 'ATX' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if '전원부' in column_name:
                    tfidv_df[column_name] = item + 0.008
                if '그래픽' in column_name:
                    tfidv_df[column_name] = item + 0.006
                if 'PCI' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if 'M2' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if 'SATA' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if 'LAN' in column_name:
                    tfidv_df[column_name] = item + 0.002
        if prod_id == "gpulist":
            for column_name, item in tfidv_df.iteritems():
                if 'MHz' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if 'GDDR' in column_name:
                    tfidv_df[column_name] = item + 0.008
                if 'GB' in column_name:
                    tfidv_df[column_name] = item + 0.006
                if '개' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if '개 팬' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if 'PCle' in column_name:
                    tfidv_df[column_name] = item + 0.003
                if '년' in column_name:
                    tfidv_df[column_name] = item + 0.002
        if prod_id == "memorylist":
            for column_name, item in tfidv_df.iteritems():
                if 'DDR' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if 'MHz' in column_name:
                    tfidv_df[column_name] = item + 0.008
                if '램' in column_name:
                    tfidv_df[column_name] = item + 0.006
                if 'V' in column_name:
                    tfidv_df[column_name] = item + 0.004
        if prod_id == "ssdlist":
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
                if '년' in column_name:
                    tfidv_df[column_name] = item + 0.002
        if prod_id == "hddlist":
            for column_name, item in tfidv_df.iteritems():
                if '인치' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if 'TB' in column_name:
                    tfidv_df[column_name] = item + 0.008
                if 'tb' in column_name:
                    tfidv_df[column_name] = item + 0.008
                if 'SATA' in column_name:
                    tfidv_df[column_name] = item + 0.006
                if 'RPM' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if 'rpm' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if '두께' in column_name:
                    tfidv_df[column_name] = item + 0.003
                if '년' in column_name:
                    tfidv_df[column_name] = item + 0.002
                if '무게' in column_name:
                    tfidv_df[column_name] = item + 0.001
        if prod_id == "powerlist":
            for column_name, item in tfidv_df.iteritems():
                if 'W' in column_name:
                    tfidv_df[column_name] = item + 0.01
                if '개' in column_name:
                    tfidv_df[column_name] = item + 0.006
                if '핀' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if 'V' in column_name:
                    tfidv_df[column_name] = item + 0.004
                if '년' in column_name:
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
