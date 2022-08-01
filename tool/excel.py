import xlsxwriter
from pathlib import Path, PurePosixPath
from tool.sqlite import check

# excel文件路径
excel_file = PurePosixPath(Path(__file__).parent.parent, 'resources/record/识别记录.xlsx')


def save_excel():
    sql = 'select * from records'
    infos = check(sql)
    workbook = xlsxwriter.Workbook(excel_file)
    worksheet1 = workbook.add_worksheet('sheet1')
    worksheet1.activate()
    # 设置水平居中、垂直居中
    str_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    title = ['序号', '花朵名称', 'CNN模型', '准确率','图片来源', '识别时间', '花朵信息']
    # 设置列宽
    worksheet1.set_column(0, 1, 10)
    worksheet1.set_column(2, 2, 15)
    worksheet1.set_column(3, 3, 10)
    worksheet1.set_column(4, 4, 15)
    worksheet1.set_column(5, 5, 20)
    worksheet1.set_column(6, 6, 50)
    # 写入内容
    worksheet1.write_row('A1', title, str_format)
    for i, data in enumerate(infos):
        row = 'A' + str(i + 2)
        worksheet1.write_row(row, data, str_format)
    try:
        workbook.close()
        msg = f'成功导出{len(infos)}条识别记录到resources/record/识别记录.xlsx中！'
    except:
        msg = '导出识别记录失败，请先关闭文件再进行导出操作！'
    print(msg)
    return msg


if __name__ == '__main__':
    save_excel()
