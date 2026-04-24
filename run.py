#!/usr/bin/env python3
import os
import shutil
import sys
import argparse
from pathlib import Path

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Copy prompt2.txt and rules.txt files from subfolders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python advanced_file_copier.py sectors /tmp/output
  python advanced_file_copier.py ./folders ./output --files prompt.txt rules.txt
  python advanced_file_copier.py ./data ./backup --prefix backup_ --verbose
        """
    )
    
    parser.add_argument('source', help='Source folder containing subfolders')
    parser.add_argument('target', help='Target folder for copied files')
    parser.add_argument('--files', nargs='+', default=['prompt2.txt', 'rules.txt'],
                       help='Files to copy (default: prompt2.txt rules.txt)')
    parser.add_argument('--prefix', default='', 
                       help='Prefix for copied files (default: subfolder_name)')
    parser.add_argument('--suffix', default='', 
                       help='Suffix for copied files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be copied without actually copying')
    
    return parser

def copy_files_advanced(source_folder, target_folder, files_to_copy, prefix, suffix, verbose, dry_run):
    """Advanced file copying with options"""
    
    if not dry_run and not os.path.exists(target_folder):
        os.makedirs(target_folder)
        if verbose:
            print(f"📁 Created target folder: {target_folder}")

    results = {
        "copied_files": [],
        "skipped_folders": [],
        "errors": []
    }

    if verbose or dry_run:
        print(f"\n🔍 Scanning subfolders in: {source_folder}")
        print(f"📋 Looking for files: {', '.join(files_to_copy)}")
        if dry_run:
            print("🧪 DRY RUN MODE - No files will be copied")
        print("-" * 50)

    # Process each subfolder
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        if os.path.isdir(item_path):
            try:
                if verbose or dry_run:
                    print(f"📂 Processing: {item}")
                
                files_found = 0
                
                # Process each specified file
                for filename in files_to_copy:
                    source_file = os.path.join(item_path, filename)
                    
                    if os.path.exists(source_file):
                        # Generate target filename
                        base_name = Path(filename).stem
                        extension = Path(filename).suffix
                        
                        if prefix:
                            target_filename = f"{prefix}{base_name}{suffix}{extension}"
                        else:
                            target_filename = f"{item}_{base_name}{suffix}{extension}"
                        
                        target_path = os.path.join(target_folder, target_filename)
                        
                        if dry_run:
                            print(f"  📄 Would copy: {filename} → {target_filename}")
                        else:
                            shutil.copy2(source_file, target_path)
                            if verbose:
                                print(f"  ✅ {filename} → {target_filename}")
                        
                        results["copied_files"].append({
                            "source": source_file,
                            "target": target_path,
                            "type": filename
                        })
                        files_found += 1
                    else:
                        if verbose or dry_run:
                            print(f"  ⚠️  {filename} not found")
                
                if files_found == 0:
                    results["skipped_folders"].append(item)
                    if verbose or dry_run:
                        print(f"  ❌ No specified files found")
                
                if verbose or dry_run:
                    print()
                
            except Exception as e:
                results["errors"].append({
                    "folder": item,
                    "error": str(e)
                })
                print(f"  ❌ Error processing {item}: {str(e)}")

    return results

def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate source folder
    if not os.path.exists(args.source):
        print(f"❌ Error: Source folder '{args.source}' does not exist!")
        sys.exit(1)
    
    if not os.path.isdir(args.source):
        print(f"❌ Error: '{args.source}' is not a directory!")
        sys.exit(1)
    
    print("🚀 Advanced File Copier")
    print(f"📁 Source: {args.source}")
    print(f"📁 Target: {args.target}")
    
    # Copy files
    results = copy_files_advanced(
        args.source, 
        args.target, 
        args.files, 
        args.prefix, 
        args.suffix, 
        args.verbose, 
        args.dry_run
    )
    
    # Print summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"📊 Files {'would be copied' if args.dry_run else 'copied'}: {len(results['copied_files'])}")
    print(f"📂 Folders skipped: {len(results['skipped_folders'])}")
    print(f"❌ Errors: {len(results['errors'])}")
    
    if not args.dry_run and results['copied_files']:
        print("✅ Operation completed successfully!")

if __name__ == '__main__':
    main()

