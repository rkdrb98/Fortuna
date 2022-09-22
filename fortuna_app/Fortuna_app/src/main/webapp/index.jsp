<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ page import="javaPro.UserDAO"%>
<%@ page import="javaPro.UserDTO"%>
<%@ page import="java.util.ArrayList"%>




<link href="./css/login_test_css.css" rel="stylesheet">
<link href="./css/style_test_css.css" rel="stylesheet">

<div class="login_containers">


	<div>
		<h2>로그인</h2>
	</div>

	<form method="post" action="./login/login_action.jsp">
		<h3>아이디</h3>
		<div class="loginID">
			<input type="text" class="input" placeholder="아이디" name="userID"
				maxlength="20">
		</div>
		<h3>비밀번호</h3>
		<div class="loginPassword">
			<input type="password" class="input" placeholder="비밀번호"
				name="userPassword" maxlength="20">
		</div>
		<input type="submit" class="bt_login" value="로그인"> <input
			type="button" class="bt_login" value="회원가입"
			onClick="location.href ='./member/join_test.jsp'">

	</form>

</div>



<footer>
	<div class="footer_container">

		<div class="footB">Copyright Fortuna Rights Reserved.</div>
	</div>
</footer>
</html>
