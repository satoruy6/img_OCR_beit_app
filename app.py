# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

st.set_page_config(
  page_title="img_OCR_beit app",
  page_icon="🚁",
)

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.write("BEITベースの画像認識モデルを使って何の画像かを判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        #results = predict(img)
        classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
        results = classifier(img)

        # 結果の表示
        st.subheader("判定結果")
        n_top = 3  # 確率が高い順に3位まで返す
        for result in results[:n_top]:
            st.write(str(round(result["score"]*100, 2)) + "%の確率で" + result["label"] + "です。")

        # 円グラフの表示
        #pie_labels = [result["label"] for result in results[:n_top]]
        #pie_labels.append("others")
        #pie_probs = [result["score"] for result in results[:n_top]]
        #pie_probs.append(sum([result["score"] for result in results[n_top:]]))
        #fig, ax = plt.subplots()
        #wedgeprops={"width":0.3, "edgecolor":"white"}
        #textprops = {"fontsize":6}
        #ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
        #       textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        #st.pyplot(fig)
