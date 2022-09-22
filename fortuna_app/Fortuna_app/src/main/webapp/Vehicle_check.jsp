<%@ page language="java" contentType="text/html; charset=EUC-KR"
	pageEncoding="EUC-KR"%>
<%@ page import="javaPro.inquiryDAO"%>
<%@ page import="javaPro.inquiry"%>
<%@ page import="java.util.ArrayList"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="EUC-KR">
<title>적발 차량 조회</title>
<head>
<%@ include file="header.jsp"%>

</head>

<style>
table {
	margin: 0 auto;
	text-align: center;
	border-collapse: collapse;
	border: 1px solid #d4d4d4;
}

tr:nth-child(even) {
	background: #d4d4d4;
}

th, td {
	padding: 10px 30px;
}

th {
	border-bottom: 1px solid #d4d4d4;
}
</style>


</head>
<body>
	<%
	int pageNumber = 1; // 기본 페이지
	if (request.getParameter("pageNumber") != null) {
		pageNumber = Integer.parseInt(request.getParameter("pageNumber"));
	}
	%>



	<div class="container">
		<div class="row">
			<table class="table table-striped"
				style="text-align: center; border: 1px solid #dddddd">
				<thead>
					<tr>
						<th style="background-color: #eeeeee; text-align: center;">번호</th>
						<th style="background-color: #eeeeee; text-align: center;">시간</th>
						<th style="background-color: #eeeeee; text-align: center;">위반내용</th>
						<th style="background-color: #eeeeee; text-align: center;">적발장소</th>
						<th style="background-color: #eeeeee; text-align: center;">차량속도</th>
						<th style="background-color: #eeeeee; text-align: center;">제한속도</th>
					</tr>
				</thead>
				<tbody>
					<%
					inquiryDAO InquiryDAO = new inquiryDAO();
					ArrayList<inquiry> list = InquiryDAO.getlist(pageNumber);

					for (int i = 0; i < list.size(); i++) {
					%>
					<tr>
						<td><%=list.get(i).getinquiryID()%></td>
						<td><%=list.get(i).getDate()%></td>
						<td><%=list.get(i).getVehicle_type()%></td>
						<td><%=list.get(i).getregion_name()%></td>
						<td><%=list.get(i).getvehicle_speed()%></td>
						<td><%=list.get(i).getspeed_limit()%></td>

					</tr>
					<%
					}
					%>
				</tbody>
			</table>
			<%
			if (pageNumber != 1) {
			%>
			<a href="Inquiry.jsp?pageNumber=<%=pageNumber - 1%>"
				class="btn btn-success btn-arraw-left">이전</a>
			<%
			}
			if (InquiryDAO.nextPage(pageNumber + 1)) {
			%>
			<a href="Inquiry.jsp?pageNumber=<%=pageNumber + 1%>"
				class="btn btn-success btn-arraw-left">다음</a>
			<%
			}
			%>

		</div>
	</div>

</body>
</html>