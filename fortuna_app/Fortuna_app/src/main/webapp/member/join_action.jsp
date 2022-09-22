<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ page import="javaPro.UserDAO" %>
<%@ page import="java.io.PrintWriter" %>
<% request.setCharacterEncoding("UTF-8"); %>

<!DOCTYPE html>

<html lang="ko">
    <head>
        <meta charset="UTF-8">
        
        <title>
            2조 로그인 ACTION
        </title>
    </head>
    
    <body>
        <%
        String userID		= request.getParameter("userID");
        String userPassword = request.getParameter("userPassword");

        
     	// -1: 서버 오류 / 0: 이미존재하는 아이디 / 1: 성공
        if(userID == null || userPassword == null) {
        	PrintWriter script = response.getWriter();
        	script.println("<script>");
        	script.println("alert('빈칸을 확인해 주세요')");
        	script.println("history.back()");
        	script.println("</script>");
        } else {
        	UserDAO userDAO = new UserDAO();
        	
        	userDAO.setUserID(userID);
        	userDAO.setUserPassword(userPassword);

        	
            int result = userDAO.join(userDAO);
            	
            if(result == -1) {
            	PrintWriter script = response.getWriter();
            	script.println("<script>");
            	script.println("alert('서버오류')");
            	script.println("history.back()");
            	script.println("</script>");
            } else if(result == 0) {
            	PrintWriter script = response.getWriter();
            	script.println("<script>");
            	script.println("alert('이미 아이디가 존재합니다.')");
            	script.println("history.back()");
            	script.println("</script>");
            } else {
            	PrintWriter script = response.getWriter();
            	script.println("<script>");
            	script.println("alert('회원가입을 축하드립니다.')");
            	script.println("location.href = '../index.jsp'");
            	script.println("</script>");
            }
        }
        %>
    </body>
    
</html>