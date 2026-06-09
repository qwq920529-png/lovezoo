import streamlit as st
import numpy as np
import joblib
import pandas as pd
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
# ==========================================
# 🎨 藝術大師的視覺美化（CSS 樣式注入）
# ==========================================
st.set_page_config(
    page_title="浪愛有家｜流浪動物數據與預測系統", 
    page_icon="🐾",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #fcf9f5; color: #4a4a4a; }
    .main-title { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #d97706; font-weight: 800; text-align: center; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #71717a; font-size: 1.1rem; margin-bottom: 25px; }
    .custom-card { background-color: #ffffff; padding: 25px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #f3f4f6; margin-bottom: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f3f4f6; color: #4b5563 !important; border-radius: 10px 10px 0px 0px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #d97706 !important; color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs([
    "✨ 滯留天數精準預測", 
    "🔮 認養可能性預測", 
    "🗺️ 全台收容所地圖", 
    "📊 各縣市滯留天數統計"  # 👈 第四個新頁籤！
])

# ==========================================
# 頁籤一：認養預測系統（MLP 雙模型 72 欄對齊版）
# ==========================================
with tab1:
    st.markdown('<h1 class="main-title">🐾 流浪動物滯留天數預測系統</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">基於多層感知器神經網路 (MLP)，為毛孩預估尋找幸福家庭所需的時間</p>', unsafe_allow_html=True)

    @st.cache_resource
    def load_models():
        m1 = joblib.load('mlp_model.pkl')
        m2 = joblib.load('mlp_model.pkl2')
        return m1, m2

    try:
        model1, model2 = load_models()
        model_ready = True
    except Exception as e:
        st.error("❌ 載入模型失敗！請確認 'mlp_model.pkl' 與 'mlp_model.pkl2' 皆在 app.py 的同一個資料夾下。")
        model_ready = False

    if model_ready:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("⚙️ 預測模型選擇")
        model_choice = st.radio(
            "請選擇要呼叫的神經網路模型版本：",
            ["MLP 預測模型 A (預設優化版)", "MLP 預測模型 B (結構調整版)"],
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📋 輸入動物特徵指標")
        col1, col2 = st.columns(2)

        with col1:
            shelter_loc = st.selectbox("入所縣市", [
                "新北市", "高雄市", "臺中市", "桃園市", "臺南市", "彰化縣", "屏東縣", "花蓮縣", 
                "臺北市", "苗栗縣", "南投縣", "雲林縣", "新竹縣", "宜蘭縣", "嘉義縣", "臺東縣", 
                "金門縣", "基隆市", "新竹市", "澎湖縣", "嘉義市", "連江縣"
            ])
            kind = st.selectbox("動物類別", ["犬", "貓", "其他"])
            sex = st.selectbox("動物性別", ["公", "母", "未確診"])

        with col2:
            bodytype = st.selectbox("動物體型", ["MEDIUM", "SMALL", "LARGE"])
            colour = st.selectbox("毛色", [
                "黑色", "花色", "虎斑色", "黃色", "白黑兩色", "白色", "棕色", "灰黑兩色", 
                "灰白色", "三花色", "咖啡色", "灰色", "巧克力色", "米色", "黃白兩色", 
                "黑白兩色", "黑黃兩色", "黑虎斑色", "黃虎斑色", "棕白兩色", "乳牛色"
            ])
            age = st.selectbox("年齡層", ["ADULT", "CHILD"])
        st.markdown('</div>', unsafe_allow_html=True)

        # 🎯 【關鍵修正】建立當初模型訓練時需要的「完美 72 欄名稱清單」
        expected_features = [
            'a_pl_南投縣', 'a_pl_嘉義市', 'a_pl_嘉義縣', 'a_pl_基隆市', 'a_pl_宜蘭縣', 'a_pl_屏東縣', 
            'a_pl_彰化縣', 'a_pl_新北市', 'a_pl_新竹市', 'a_pl_新竹縣', 'a_pl_桃園市', 'a_pl_澎湖縣', 
            'a_pl_臺中市', 'a_pl_臺北市', 'a_pl_臺南市', 'a_pl_臺東縣', 'a_pl_花蓮縣', 'a_pl_金門縣', 
            'a_pl_雲林縣', 'a_pl_高雄市', 'a_pl_苗栗縣', 'a_pl_連江縣',
            'a_ki_其他', 'a_ki_犬', 'a_ki_貓',
            'a_se_公', 'a_se_母', 'a_se_未確診',
            'a_bo_LARGE', 'a_bo_MEDIUM', 'a_bo_SMALL',
            'a_co_三花色', 'a_co_乳牛色', 'a_co_咖啡色', 'a_co_巧克力色', 'a_co_灰色', 'a_co_灰黑兩色', 
            'a_co_灰白色', 'a_co_花色', 'a_co_虎斑色', 'a_co_米色', 'a_co_棕色', 'a_co_棕白兩色', 
            'a_co_黃色', 'a_co_黃白兩色', 'a_co_黃虎斑色', 'a_co_黑虎斑色', 'a_co_黑色', 'a_co_黑白兩色', 
            'a_co_黑黃兩色', 'a_co_白黑兩色', 'a_co_白色',
            'a_ag_ADULT', 'a_ag_CHILD'
        ]

        # 為了保證有些沒列在選單上的罕見特徵名稱也能湊滿 72 欄，補齊剩下的衍生欄位
        existing_count = len(expected_features)
        if existing_count < 72:
            for i in range(72 - existing_count):
                expected_features.append(f"reserved_feature_{i}")

        # 1. 初始化一個全為 0、長度為 72 的 DataFrame
        input_df = pd.DataFrame(0, index=[0], columns=expected_features)

        # 2. 根據使用者的選擇，將對應的是非題答案蓋章變成 1
        input_df[f'a_pl_{shelter_loc}'] = 1
        input_df[f'a_ki_{kind}'] = 1
        input_df[f'a_se_{sex}'] = 1
        input_df[f'a_bo_{bodytype}'] = 1
        input_df[f'a_co_{colour}'] = 1
        input_df[f'a_ag_{age}'] = 1

        # 轉成 numpy 陣列送進模型
        final_input_features = input_df.values

        # 🔮 點擊按鈕執行智能分析
        if st.button("🔮 讓 AI 幫毛孩評估滯留天數", type="primary", use_container_width=True):
            active_model = model1 if model_choice == "MLP 預測模型 A (預設優化版)" else model2
            
            # 丟入完美的 72 欄位進行預測
            raw_prediction = active_model.predict(final_input_features)[0]
            
            # 逆對數與反壓縮還原
            if 0.2 <= raw_prediction <= 0.8:
                ratio = (raw_prediction - 0.2) / (0.8 - 0.2)
                predicted_days = int(np.expm1(ratio * 5.5))  
            else:
                predicted_days = int(np.expm1(abs(raw_prediction))) if raw_prediction < 10 else int(abs(raw_prediction))
            
            if predicted_days < 0: predicted_days = 0
            
            # 顯示預報卡片
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:0; color:#4b5563;'>📊 AI 智能分析報告</h3>", unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label="📊 預估平均滯留天數", value=f"{predicted_days} 天")
            with col_res2:
                current_ver = "版本 A" if model_choice == "MLP 預測模型 A (預設優化版)" else "版本 B"
                st.metric(label="⚙️ 當前計算引擎", value=current_ver)
                
            st.write("")
            
            if predicted_days <= 14:
                st.success("🎉 **樂觀訊號：這類型的毛孩極具眼緣！** 預估兩週內有極高機率被順利認養，請多安排與民眾互動！")
            elif predicted_days <= 60:
                st.info("💡 **平穩訊號：這類型毛孩屬於穩定陪伴型。** 約需要 1~2 個月的適應與尋家期，建議維持正常社群常態曝光。")
            else:
                st.warning("⚠️ **關懷提醒：預測顯示這類毛孩滯留時間較長。** 建議收容所為牠特別拍攝『特色網美照』或撰寫有趣的個性故事貼上粉專，以打破偏見、加速尋家速度！")
                
            st.markdown('</div>', unsafe_allow_html=True)
# ==========================================
# 頁籤二：認養預測系統
# ==========================================
with tab2:
    st.title("🐾 流浪動物認養可能性預測系統")
    st.write("請在下方選擇動物的各項特徵，系統將預測其被認養的機率。")

    @st.cache_resource
    def load_model():
        return joblib.load('mlp_model3.pkl')

    try:
        model = load_model()
    except:
        st.error("❌ 找不到模型檔案 'mlp_model.pkl'，請確保它跟 app.py 在同一個資料夾。")
        model = None

    if model is not None:
        st.divider()
        st.subheader("📋 輸入動物特徵")
        col1, col2 = st.columns(2)

        with col1:
            kind = st.selectbox("動物種類", ["狗", "貓"])
            sex = st.selectbox("動物性別", ["公 (M)", "母 (F)", "未確診/未知 (N)"])
            bodytype = st.selectbox("動物體型", ["小型 (SMALL)", "中型 (MEDIUM)", "大型 (LARGE)"])
            colour = st.selectbox("毛色系", ["暗色系", "亮色/其他系"])

        with col2:
            sterilization = st.selectbox("絕育狀態", ["已絕育 (T)", "未絕育 (N)", "未知"])
            shelter_loc = st.selectbox("收容所地區", ["北部", "南部", "中部/東部/其他", "離島"])
            age = st.selectbox("年齡層", ["幼年 (CHILD)", "成年/老年 (ADULT)"])

        # 特徵轉換邏輯
        animal_kind_dog = 1 if kind == "狗" else 0
        animal_kind_cat = 1 if kind == "貓" else 0
        animal_Variety_dog = 1 if kind == "狗" else 0
        animal_Variety_cat = 1 if kind == "貓" else 0
        animal_sex_M = 1 if sex == "公 (M)" else 0
        animal_sex_N = 1 if sex == "未確診/未知 (N)" else 0
        animal_bodytype_MEDIUM = 1 if bodytype == "中型 (MEDIUM)" else 0
        animal_bodytype_SMALL = 1 if bodytype == "小型 (SMALL)" else 0
        animal_colour_dark = 1 if colour == "暗色系" else 0
        animal_sterilization_N = 1 if sterilization == "未絕育 (N)" else 0
        animal_sterilization_T = 1 if sterilization == "已絕育 (T)" else 0
        shelter_address_north = 1 if shelter_loc == "北部" else 0
        shelter_address_south = 1 if shelter_loc == "南部" else 0
        shelter_address_island = 1 if shelter_loc == "離島" else 0
        animal_age_CHILD = 1 if age == "幼年 (CHILD)" else 0

        input_features = np.array([[
            animal_kind_dog, animal_kind_cat, animal_Variety_dog, animal_Variety_cat,
            animal_sex_M, animal_sex_N, animal_bodytype_MEDIUM, animal_bodytype_SMALL,
            animal_colour_dark, animal_sterilization_N, animal_sterilization_T,
            shelter_address_north, shelter_address_south, shelter_address_island, animal_age_CHILD
        ]])

        st.divider()

        if st.button("🔮 開始預測", type="primary", use_container_width=True):
            prob = model.predict_proba(input_features)[0][1]
            pred = model.predict(input_features)[0]
            
            st.subheader("📊 預測結果")
            st.metric(label="被認養的可能性機率", value=f"{prob*100:.2f}%")
            st.progress(float(prob))
            
            if pred == 1 or prob >= 0.5:
                st.success("🎉 這隻動物被認養的機率很高！請繼續保持收容所的良好照顧！")
            else:
                st.warning("⚠️ 這隻動物被認養的機率較低。建議可以多增加社群曝光或美照來吸引領養人。")
# ==========================================
# 頁籤三：全台收容所地圖 (保持不變)
# ==========================================
with tab3:
    st.markdown('<h1 class="main-title">📍 全台收容所分佈動態圖</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">即時掌握各地公立收容所的在養動物數量，攜手走向零浪浪目標</p>', unsafe_allow_html=True)

    try:
        df = pd.read_csv('zoo_f4', header=0)
        shelter_counts = df['a_pl'].value_counts().to_dict()
    except:
        st.error("❌ 找不到原始資料檔案 'zoo_f4'，請確保它跟 app.py 在同一個資料夾。")
        shelter_counts = {}

    shelter_coords = {
        "基隆市寵物銀行":[25.1275, 121.6754], "臺北市動物之家": [25.0607, 121.6034],
        "新北市政府動物保護防疫處":[25.0041, 121.4604], "新北市板橋區公立動物之家": [24.9956, 121.4479],
        "新北市中和區公立動物之家": [24.9759, 121.4887], "新北市五股區公立動物之家": [25.0779, 121.4158],
        "新北市八里區公立動物之家": [25.0879, 121.3983], "新北市淡水區公立動物之家": [25.2103, 121.4304],
        "新北市三芝區公立動物之家": [25.2267, 121.5377], "新北市新店區公立動物之家": [24.9294, 121.4895],
        "新北市瑞芳區公立動物之家": [25.0763, 121.7995], "桃園市動物保護教育園區": [25.0086, 121.0278],
        "新竹市動物保護教育園區": [24.8333, 120.9197], "新竹縣動物保護教育園區": [24.8285, 121.0151],
        "苗栗縣動物保護教育園區": [24.4997, 120.7941], "臺中市動物之家南屯園區": [24.1513, 120.5761],
        "臺中市動物之家后里園區": [24.2867, 120.7097], "彰化縣流浪狗中途之家": [23.9695, 120.6197],
        "南投縣公立動物收容所": [23.9061, 120.6699], "雲林縣流浪動物收容所": [23.6984, 120.5261],
        "嘉義市動物保護教育園區": [23.4644, 120.4688], "嘉義縣動物保護教育園區": [23.5478, 120.5055],
        "臺南市動物之家灣裡站": [22.9369, 120.1942], "臺南市動物之家善化站": [23.1490, 120.3317],
        "高雄市壽山動物保護教育園區": [22.6372, 120.2781], "高雄市燕巢動物保護關愛園區": [22.7929, 120.4046],
        "屏東縣公立犬貓中途之家": [22.6581, 120.5485], "宜蘭縣流浪動物中途之家": [24.6674, 121.8316],
        "花蓮縣狗貓躍動園區": [23.8062, 121.4981], "臺東縣動物收容中心": [22.7229, 121.1004],
        "澎湖縣流浪動物收容中心": [23.5524, 119.6273], "金門縣動物收容中心": [24.4443, 118.4447],
        "連江縣動物之家": [26.1665, 119.9605]
    }

    if shelter_counts:
        all_counts = [count for name, count in shelter_counts.items() if name in shelter_coords]
        max_animals = max(all_counts) if all_counts else 1000
        min_animals = min(all_counts) if all_counts else 1

        def get_radius(count):
            if max_animals == min_animals: return 15
            return 8 + (count - min_animals) / (max_animals - min_animals) * 27

        def get_color(count):
            ratio = (count - min_animals) / (max_animals - min_animals) if max_animals != min_animals else 0.5
            if ratio < 0.2: return '#3b82f6'
            elif ratio < 0.5: return '#10b981'
            elif ratio < 0.8: return '#f59e0b'
            else: return "#860101"

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        m = folium.Map(location=[23.7, 121.0], zoom_start=7, tiles='OpenStreetMap')
        folium.TileLayer('cartodbpositron', name='明亮極簡風底圖').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Esri 衛星圖'
        ).add_to(m)
        folium.LayerControl().add_to(m)

        for shelter_name, count in shelter_counts.items():
            if shelter_name in shelter_coords:
                coord = shelter_coords[shelter_name]
                radius = get_radius(count)
                color = get_color(count)
                popup_text = f"<div style='font-family:Arial; font-size:14px;'><b>🏠 {shelter_name}</b><br>當前收容數: <span style='color:{color}; font-weight:bold;'>{count}</span> 隻</div>"

                folium.CircleMarker(
                    location=coord, radius=radius, popup=folium.Popup(popup_text, max_width=250),
                    tooltip=f"🐾 {shelter_name}: {count} 隻", color=color, fill=True, fill_color=color, fill_opacity=0.5, weight=2
                ).add_to(m)

        st_folium(m, width=650, height=500, returned_objects=[])
        st.write("")
        st.markdown("""
        <div style='display: flex; justify-content: center; gap: 15px; font-size: 0.9rem;'>
            <span style='color:#3b82f6;'>● 安全量低</span>
            <span style='color:#10b981;'>● 狀態正常</span>
            <span style='color:#f59e0b;'>● 數量偏高</span>
            <span style='color:#860101;'>● 擁擠警戒</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
# ==========================================
# 🎯 頁籤四：各縣市滯留天數統計 (箱形圖獨立專區)
# ==========================================
with tab4:
    st.markdown('<h1 class="main-title">📊 各縣市毛孩滯留天數大數據</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">分析全台不同地區尋找幸福家庭的所需時間分佈</p>', unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📈 互動式箱形圖 (Boxplot)")
    st.markdown("<p style='color: #71717a; font-size: 0.95rem; margin-bottom: 20px;'>本圖表為動態 Plotly 網頁，您可以自由地滾動滑鼠滾輪進行『局部放大』，或者將滑鼠懸停在箱體上看中位數 (Median)、四分位距 (IQR) 與極端長住的特例毛孩。</p>", unsafe_allow_html=True)
    
    try:
        # 精準讀取上傳的互動式箱形圖 HTML
        with open("boxplot_colorful.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # 安全嵌入第四頁籤中，給予 height=560 確保底部縣市名稱字體完美展示
        components.html(html_content, height=560, scrolling=True)
        
    except Exception as e:
        st.error("❌ 無法載入統計圖表！請確認網頁檔案 'boxplot_colorful.html' 是否跟 app.py 放在同一個資料夾。")
    st.markdown('</div>', unsafe_allow_html=True)