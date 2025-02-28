import openpyxl
import weather

html=weather.get_html()
lst=weather.parse_html(html)
#创建Excel工作簿
workbook=openpyxl.Workbook()
#在Excel文件中创建工作表
sheet=workbook.create_sheet('景区天气')
#向工作表中添加数据
for item in lst:
    sheet.append(item)

workbook.save('景区天气.xlsx')
