import { parametersSchema as z, defineCustomTool } from "@roo-code/types";
import * as fs from "fs/promises";
import * as path from "path";

export default defineCustomTool({
  name: "read-file",
  description: `Reads a file from the local filesystem. Returns content with line numbers.

Usage examples:
- Single file: { "files": [{ "path": "src/app.ts" }] }
- With line ranges: { "files": [{ "path": "src/app.ts", "line_ranges": ["1-50", "100-150"] }] }
- Multiple files: { "files": [{ "path": "file1.ts" }, { "path": "file2.ts" }] }

Note: 'path' is required. 'line_ranges' is optional (1-based inclusive).`,

  parameters: z.object({
    files: z.array(z.object({
      path: z.string().describe("Path to the file to read, relative to the workspace"),
      line_ranges: z.array(z.string()).optional().describe("Optional: 1-based inclusive line ranges, e.g., ['1-50', '100-150']")
    })).describe("Array of files to read")
  }),

  async execute(args) {
    const results: string[] = [];
    const errors: string[] = [];

    // Find workspace root
    let workspaceRoot = process.env.VSCODE_CWD || process.env.INIT_CWD || process.cwd();
    
    // Search upwards for .git or .roo to find actual project root
    let currentDir = workspaceRoot;
    for (let i = 0; i < 10; i++) {
      try {
        await fs.access(path.join(currentDir, '.git'));
        workspaceRoot = currentDir;
        break;
      } catch {}
      try {
        await fs.access(path.join(currentDir, '.roo'));
        workspaceRoot = currentDir;
        break;
      } catch {}
      const parent = path.dirname(currentDir);
      if (parent === currentDir) break;
      currentDir = parent;
    }

    for (const file of args.files) {
      try {
        // Security checks
        if (path.isAbsolute(file.path)) {
          errors.push(`❌ ${file.path}: Absolute paths are not allowed. Use relative paths.`);
          continue;
        }
        
        const normalizedPath = path.normalize(file.path);
        if (normalizedPath.startsWith('..') || normalizedPath.includes('../')) {
          errors.push(`❌ ${file.path}: Access denied - cannot read files outside workspace.`);
          continue;
        }

        const fullPath = path.join(workspaceRoot, normalizedPath);
        
        let content: string;
        let lines: string[];
        let startLine = 1;
        let endLine: number;
        
        // Handle line ranges if provided
        if (file.line_ranges && file.line_ranges.length > 0) {
          const fullContent = await fs.readFile(fullPath, "utf-8");
          const allLines = fullContent.split("\n");
          
          const selectedLines: string[] = [];
          for (const range of file.line_ranges) {
            const [start, end] = range.split('-').map(Number);
            if (isNaN(start) || isNaN(end)) {
              throw new Error(`Invalid line range format: ${range}. Use 'start-end' format.`);
            }
            for (let i = start - 1; i < Math.min(end, allLines.length); i++) {
              if (i >= 0) selectedLines.push(allLines[i]);
            }
          }
          lines = selectedLines;
          startLine = 1; // Reset for display
          endLine = lines.length;
          content = lines.join('\n');
        } else {
          // Read entire file
          content = await fs.readFile(fullPath, "utf-8");
          lines = content.split("\n");
          endLine = lines.length;
        }
        
        // Format output with line numbers
        const numberedLines = lines.map((line, index) => {
          const lineNum = startLine + index;
          const paddedNum = lineNum.toString().padStart(6, ' ');
          return `${paddedNum} | ${line}`;
        }).join('\n');
        
        const totalLines = lines.length;
        const header = `File: ${file.path} (${totalLines} lines)`;
        
        results.push(`${header}\n${numberedLines}`);
        
      } catch (error: any) {
        if (error.code === 'ENOENT') {
          errors.push(`❌ ${file.path}: File not found. Attempted path: ${path.join(workspaceRoot, file.path)}`);
        } else {
          errors.push(`❌ ${file.path}: ${error.message}`);
        }
      }
    }
    
    // Build final output
    let output = '';
    if (results.length > 0) {
      output += results.join('\n\n---\n\n');
    }
    if (errors.length > 0) {
      output += (output ? '\n\n## Errors\n\n' : '') + errors.join('\n');
    }
    
    if (!output) {
      throw new Error('No files were successfully read.');
    }
    
    return output;
  }
});