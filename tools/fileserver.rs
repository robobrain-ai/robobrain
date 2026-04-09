// Robust HTTP file server for large checkpoint transfers.
// Handles 10+ GB files, concurrent connections, broken pipes.
// Usage: ./fileserver /path/to/dir [port]
// Build: rustc fileserver.rs -O -o fileserver

use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write, Read, Seek, SeekFrom};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf, Component};
use std::thread;

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = PathBuf::from(args.get(1).map(|s| s.as_str()).unwrap_or("."))
        .canonicalize().expect("Invalid directory");
    let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8766);

    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).expect("Failed to bind");
    eprintln!("Serving {} on port {}", root.display(), port);

    for stream in listener.incoming() {
        let stream = match stream { Ok(s) => s, Err(_) => continue };
        let root = root.clone();
        thread::spawn(move || { let _ = handle(stream, &root); });
    }
}

fn handle(mut stream: TcpStream, root: &Path) -> io::Result<()> {
    stream.set_write_timeout(Some(std::time::Duration::from_secs(300)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    let path_str = request_line.split_whitespace().nth(1).unwrap_or("/");
    let decoded = url_decode(path_str);
    let rel = Path::new(&decoded).strip_prefix("/").unwrap_or(Path::new(&decoded));

    // Read all headers (look for Range)
    let mut range_start: Option<u64> = None;
    let mut range_end: Option<u64> = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() { break; }
        let lower = line.to_lowercase();
        if lower.starts_with("range: bytes=") {
            let range = lower.trim_start_matches("range: bytes=").trim().to_string();
            let parts: Vec<&str> = range.split('-').collect();
            if let Some(s) = parts.get(0) { range_start = s.parse().ok(); }
            if let Some(e) = parts.get(1) { if !e.is_empty() { range_end = e.parse().ok(); } }
        }
    }

    let full_path = match resolve_path(root, rel) {
        Some(p) => p,
        None => { return send_error(&mut stream, 403, "Forbidden"); }
    };

    if full_path.is_dir() {
        return send_dir_listing(&mut stream, &full_path, path_str);
    }

    if !full_path.exists() {
        return send_error(&mut stream, 404, "Not Found");
    }

    let mut file = File::open(&full_path)?;
    let file_size = file.metadata()?.len();
    let mime = guess_mime(&full_path);

    match range_start {
        Some(start) => {
            let end = range_end.unwrap_or(file_size - 1).min(file_size - 1);
            let len = end - start + 1;
            file.seek(SeekFrom::Start(start))?;
            let header = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Type: {}\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                mime, len, start, end, file_size
            );
            stream.write_all(header.as_bytes())?;
            stream_file(&mut file, &mut stream, len)?;
        }
        None => {
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                mime, file_size
            );
            stream.write_all(header.as_bytes())?;
            stream_file(&mut file, &mut stream, file_size)?;
        }
    }
    Ok(())
}

fn stream_file(file: &mut File, stream: &mut TcpStream, mut remaining: u64) -> io::Result<()> {
    let mut buf = vec![0u8; 256 * 1024]; // 256 KB buffer
    while remaining > 0 {
        let to_read = (remaining as usize).min(buf.len());
        let n = file.read(&mut buf[..to_read])?;
        if n == 0 { break; }
        if stream.write_all(&buf[..n]).is_err() { break; } // broken pipe = just stop
        remaining -= n as u64;
    }
    Ok(())
}

fn send_error(stream: &mut TcpStream, code: u16, msg: &str) -> io::Result<()> {
    let body = format!("<h1>{} {}</h1>", code, msg);
    let resp = format!("HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", code, msg, body.len(), body);
    stream.write_all(resp.as_bytes())
}

fn send_dir_listing(stream: &mut TcpStream, dir: &Path, url_path: &str) -> io::Result<()> {
    let mut body = format!("<html><body><h1>Index of {}</h1><ul>", url_path);
    if let Ok(entries) = fs::read_dir(dir) {
        let mut names: Vec<String> = entries.filter_map(|e| e.ok()).map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let suffix = if e.path().is_dir() { "/" } else { "" };
            format!("<li><a href=\"{}/{}{}\">{}{}</a></li>", url_path.trim_end_matches('/'), name, suffix, name, suffix)
        }).collect();
        names.sort();
        for n in names { body.push_str(&n); }
    }
    body.push_str("</ul></body></html>");
    let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
    stream.write_all(resp.as_bytes())
}

fn url_decode(s: &str) -> String {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i+1]), hex_val(bytes[i+2])) {
                out.push(h * 16 + l); i += 3; continue;
            }
        }
        out.push(bytes[i]); i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn resolve_path(root: &Path, rel: &Path) -> Option<PathBuf> {
    let mut clean = PathBuf::new();
    for c in rel.components() {
        match c {
            Component::Normal(s) => clean.push(s),
            Component::CurDir => {}
            _ => return None,
        }
    }
    let full = root.join(&clean);
    if full.starts_with(root) { Some(full) } else { None }
}

fn guess_mime(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") | Some("htm") => "text/html",
        Some("json") => "application/json",
        Some("txt") => "text/plain",
        _ => "application/octet-stream",
    }
}
