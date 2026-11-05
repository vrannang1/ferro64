// Reproducible native OCR helper. Build with scripts/build-ocr.sh on macOS.
import AppKit
import PDFKit
import Vision

func recognize(_ image: CGImage) throws -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}
let args = CommandLine.arguments
if args.count < 2 { fail("Usage: ocr <input_path> [thumbnail_output_path]") }
let url = URL(fileURLWithPath: args[1])
do {
    if url.pathExtension.lowercased() == "pdf" {
        guard let document = PDFDocument(url: url) else { fail("Cannot open PDF") }
        guard document.pageCount <= 100 else { fail("PDF exceeds 100 pages; split it first") }
        var texts = [String]()
        for index in 0..<document.pageCount {
            guard let page = document.page(at: index) else { continue }
            let text = page.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let thumbnail = page.thumbnail(of: NSSize(width: 1600, height: 2200), for: .mediaBox)
            var rect = NSRect(origin: .zero, size: thumbnail.size)
            let cg = thumbnail.cgImage(forProposedRect: &rect, context: nil, hints: nil)
            if index == 0 && args.count > 2, let cg = cg {
                let rep = NSBitmapImageRep(cgImage: cg)
                if let png = rep.representation(using: .png, properties: [:]) {
                    try png.write(to: URL(fileURLWithPath: args[2]))
                }
            }
            let extracted = text.isEmpty ? (try cg.map { try recognize($0) } ?? "") : text
            texts.append("--- Page \(index + 1) ---\n\(extracted)")
        }
        print(texts.joined(separator: "\n\n"))
    } else {
        guard let image = NSImage(contentsOf: url) else { fail("Cannot open image") }
        var rect = NSRect(origin: .zero, size: image.size)
        guard let cg = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else { fail("Cannot decode image") }
        print(try recognize(cg))
    }
} catch { fail(error.localizedDescription) }
