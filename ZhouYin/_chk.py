s=open('ae_gui_qt.py',encoding='utf-8').read()
# find the w=QWidget() that follows QMessageBox.information about 报告已保存
i = s.find('报告已保存')
print(repr(s[i:i+300]))
