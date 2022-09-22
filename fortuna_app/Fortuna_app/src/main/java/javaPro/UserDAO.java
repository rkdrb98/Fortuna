package javaPro;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

public class UserDAO {
	private String userID;
	private String userPassword;


	private Connection con;
	private ResultSet rs;
	private static UserDAO instance =new UserDAO();

	public UserDAO() {
		try {
			String dbURL = "jdbc:mariadb://rkdrb0070.cafe24.com/rkdrb0070";
			String dbID = "rkdrb0070";
			String dbPassword = "Krkdrb0070";
            Class.forName("org.mariadb.jdbc.Driver");
			con = DriverManager.getConnection(dbURL, dbID, dbPassword);
		} catch (Exception e) {
			e.printStackTrace();

		}
	}

	public int login(String userID, String userPassword) {
		try {
			PreparedStatement pst = con.prepareStatement("SELECT userPassword FROM user WHERE userID = ?");
			pst.setString(1, userID);
			rs = pst.executeQuery();
			if (rs.next()) {
				return rs.getString(1).equals(userPassword) ? 1 : 0;
			} else {
				return -2;
			}
		} catch (Exception e) {
			e.printStackTrace();
			return -1;
		}
	}

	
	public boolean ID_Check(String userID) {
		try {
			PreparedStatement pst = con.prepareStatement("SELECT * FROM user WHERE userID = ?");
			pst.setString(1, userID);
			rs = pst.executeQuery();
			if (rs.next()) {
				return false;
			} else {
				return true;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return false;
	}

	public void deleteMember(String id)throws Exception{
		        Connection con=null;
		      PreparedStatement pstmt =null;
		        String sql = null;
		        try{
		        	String dbURL = "jdbc:mariadb://localhost:3306/rkdrb0070";
					String dbID = "rkdrb0070";
					String dbPassword = "Krkdrb0070";
					Class.forName("com.mysql.jdbc.Driver");
					con = DriverManager.getConnection(dbURL, dbID, dbPassword);
					
		            sql = "delete from user where userID=?";
		            pstmt =con.prepareStatement(sql);
		          pstmt.setString(1, id);
		         pstmt.executeUpdate();
		       }catch(Exception e){
		           e.printStackTrace();
		            
		       }
		    }
	public static UserDAO getInstance(){
        return instance;
    }
public int userCheck(String id, String passwd)throws Exception{
        
	String dbURL = "jdbc:mysql://localhost/rkdrb0070";
	String dbID = "rkdrb0070";
	String dbPassword = "Krkdrb0070!";
	Class.forName("com.mysql.jdbc.Driver");
	con = DriverManager.getConnection(dbURL, dbID, dbPassword);
	
       
        PreparedStatement pstmt = null;
        ResultSet rs =null;
        String sql="";
        String dbpasswd="";
        int x = -1;
        
        try{
            
            sql ="select userPassword from user where userID = ?";
            pstmt =con.prepareStatement(sql);
            pstmt.setString(1, id);
            rs=pstmt.executeQuery();
            
            if(rs.next()){
                dbpasswd =rs.getString("userPassword");
                System.out.println(dbpasswd);
                if(dbpasswd.equals(passwd))
                    x=1; //인증성공
                else
                    x=0; //비밀번호 틀림
            }else
                x=-1; //해당 아이디 없음
        }catch(Exception e){
            e.printStackTrace();
        }finally{
           
        }
        return x;
    }

	public int join(UserDAO userDAO) {
		if (!ID_Check(userDAO.getUserID()))
			return 0;

		try {
			PreparedStatement pst = con.prepareStatement("INSERT INTO user VALUES (?,?)");
			pst.setString(1, userDAO.getUserID());
			pst.setString(2, userDAO.getUserPassword());

			return pst.executeUpdate();
		} catch (Exception e) {
			e.printStackTrace();
			return -1;
		}
	}

	public UserDAO getUser(String userID) {
		try {
			PreparedStatement pst = con.prepareStatement("SELECT * FROM user WHERE userID = ?");
			pst.setString(1, userID);
			rs = pst.executeQuery();
			if (rs.next()) {
				UserDAO userDAO = new UserDAO();
				userDAO.setUserID(rs.getString(1));
				userDAO.setUserPassword(rs.getString(2));
				
				return userDAO;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public String getUserID() {
		return userID;
	}

	public void setUserID(String userID) {
		this.userID = userID;
	}

	public String getUserPassword() {
		return userPassword;
	}

	public void setUserPassword(String userPassword) {
		this.userPassword = userPassword;
	}}

	