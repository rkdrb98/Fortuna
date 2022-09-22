<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<% 
   if(session.getAttribute("userID") == null){
       response.sendRedirect("../index.jsp");
   }else{
%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head><title>회원탈퇴</title>
</head>
<body>
<form name="myform" action="delete_member.jsp" method="post">
아이디 : <input type=password name="userID"  size="15" maxlength="12"><br/>
비밀번호 : <input type=password name="userPassword"  size="15" maxlength="12"><br/>
      
<input type=submit value="회원탈퇴"> 
<input type="button" value="취  소" onclick="location.href='../index.jsp'">
</form>
</body>
</html>
<%}%>