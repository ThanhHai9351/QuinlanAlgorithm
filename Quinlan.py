import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('Data.csv')  # Thay 'ten_file.csv' bằng tên tệp CSV của bạn


label_encoder = LabelEncoder()
data['Ten'] = label_encoder.fit_transform(data['Ten'])
data['Toc'] = label_encoder.fit_transform(data['Toc'])
data['Chieu_Cao'] = label_encoder.fit_transform(data['Chieu_Cao'])
data['Can_Nang'] = label_encoder.fit_transform(data['Can_Nang'])
data['Dung_Kem'] = label_encoder.fit_transform(data['Dung_Kem'])
data['Ket_Qua'] = label_encoder.fit_transform(data['Ket_Qua'])

# Chia dữ liệu thành features (đặc trưng) và labels (nhãn)
X = data.drop('Toc', axis=1)  # Thay 'target_column' bằng tên cột nhãn của bạn
X = data.drop('Chieu_Cao', axis=1) 
X = data.drop('Can_Nang', axis=1) 
X = data.drop('Dung_Kem', axis=1) 
y = data.drop('Ket_Qua', axis=1) 
y = data['Ket_Qua']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo một bộ phân loại cây quyết định ID3
clf = DecisionTreeClassifier(criterion='entropy')  # Sử dụng entropy cho ID3

# Đào tạo bộ phân loại trên dữ liệu huấn luyện
clf.fit(X_train, y_train)

# Tiến hành dự đoán trên dữ liệu kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá độ chính xác của bộ phân loại
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)

# Vẽ cây quyết định
dot_data = export_graphviz(clf, out_file=None,
                feature_names=X.columns,  
                class_names=[str(c) for c in y.unique()],  # Chuyển đổi các nhãn thành chuỗi
                filled=True, rounded=True, special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('decision_tree.png')  # Lưu cây quyết định vào một tệp PNG
Image(graph.create_png())
