package javaPro;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;

public class inquiryDAO {

	private Connection conn;
	private ResultSet rs;

	public inquiryDAO() {
		try {
			String dbURL = "jdbc:mariadb://localhost:3306/rkdrb0070";
			String dbID = "rkdrb0070";
			String dbPassword = "Krkdrb0070";
			Class.forName("org.mariadb.jdbc.Driver");
			conn = DriverManager.getConnection(dbURL, dbID, dbPassword);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public boolean nextPage(int pageNumber) {
		String SQL = "SELECT * FROM inquiry WHERE ID < ? AND Available = 1 ORDER BY ID DESC LIMIT 10";
		try {
			PreparedStatement pstmt = conn.prepareStatement(SQL);
			pstmt.setInt(1, getNext() - (pageNumber - 1) * 10);
			rs = pstmt.executeQuery();
			if (rs.next()) {
				return true;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return false;
	}

	public ArrayList<inquiry> getlist(int pageNumber) {
		String SQL = "SELECT * FROM inquiry WHERE ID < ? AND Available = 1 ORDER BY ID DESC LIMIT 10";
		ArrayList<inquiry> list = new ArrayList<inquiry>();
		try {
			PreparedStatement pstmt = conn.prepareStatement(SQL);
			pstmt.setInt(1, getNext() - (pageNumber - 1) * 10);
			rs = pstmt.executeQuery();
			while (rs.next()) {
				inquiry Inquiry = new inquiry();
				Inquiry.setinquiryID(rs.getInt(1));
				Inquiry.setVehicle_type(rs.getString(2));
				Inquiry.setregion_name(rs.getString(3));
				Inquiry.setvehicle_speed(rs.getString(4));
				Inquiry.setspeed_limit(rs.getString(5));
				Inquiry.setDate(rs.getString(6));
				Inquiry.setAvailable(rs.getInt(7));
				list.add(Inquiry);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return list;
	}

	public int delete(int ID) {
		String SQL = "UPDATE inquiry SET Available=0 WHERE ID=?";
		try {
			PreparedStatement pstmt = conn.prepareStatement(SQL);
			pstmt.setInt(1, ID);
			pstmt.executeUpdate();
			return 1;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}

	

	public int getNext() {
		String SQL = "SELECT ID FROM inquiry ORDER BY ID DESC";
		try {
			PreparedStatement pstmt = conn.prepareStatement(SQL);
			rs = pstmt.executeQuery();
			if (rs.next()) {
				return rs.getInt(1) + 1;
			}
			return 1;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}

}